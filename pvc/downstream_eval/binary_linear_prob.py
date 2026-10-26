"""Linear-probe evaluation of a frozen pretrained backbone on binary downstream tasks.

Runs k-fold cross-validation with either a random (segment-level) split or a
patient-level split, and logs F1, AUROC, and accuracy per fold and on average.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchmetrics
import argparse
import os
from collections import OrderedDict
import numpy as np
import yaml
from tabulate import tabulate
import copy
import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import numpy as np
import random
from torch.utils.data import DataLoader, BatchSampler, Sampler
import io
import concurrent.futures
from functools import partial
from tqdm import tqdm
import math
from scipy import signal
import h5py
import boto3
import io
import numpy as np
import pickle
import time
from sklearn.model_selection import StratifiedGroupKFold, KFold, StratifiedKFold
from collections import defaultdict
from downstream_eval.task_definition import *
from downstream_eval.helpers import summarize_dataset
from utils.helper_models import get_model
from utils.helper_logger import create_logger
import argparse
import logging


class HIMAE_(nn.Module):
    """Linear classifier on time-pooled bottleneck features of a frozen HiMAE encoder."""

    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = x.unsqueeze(1)
            skip_connections = []
            current_x = x
            for encoder_block in self.backbone.encoder_layers:
                current_x = encoder_block(current_x)
                skip_connections.append(current_x)
    
            bottleneck = skip_connections.pop()

        return self.classifier(bottleneck.mean(-1))


class PULSE_(nn.Module):
    """Linear classifier on the PPG embedding of a frozen PulseBridge backbone."""

    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(384, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(ppg=x,
                                    ecg=None,
                                    ids_keep_ppg=None,
                                    ids_restore_ppg=None,
                                    ids_keep_ecg=None,
                                    ids_restore_ecg=None,
                                    )
            
        return self.classifier(features['ppg_embedding'])


def print_h5_structure(h5_object, indent=0):
    """
    Recursively prints the structure of an HDF5 file or group.
    
    Args:
        h5_object: An h5py.File or h5py.Group object.
        indent (int): The current indentation level for pretty-printing.
    """
    prefix = "  " * indent
    for key, item in h5_object.items():
        if isinstance(item, h5py.Group):
            mylogger.info(f"{prefix}[GROUP] {key}")
            print_h5_structure(item, indent + 1)
        elif isinstance(item, h5py.Dataset):
            mylogger.info(f"{prefix}[DATASET] {key} | Shape: {item.shape} | Dtype: {item.dtype}")
        else:
            mylogger.info(f"{prefix}[UNKNOWN] {key}")

def _read_one_h5_from_local(bucket_name, s3_key):
    """Open an HDF5 file from a local mirror of the bucket, falling back to S3."""
    path = '../' + os.path.join(bucket_name, s3_key)
        
    if not os.path.exists(path):
        mylogger.info('Local copy not found, reading from s3...')
        return _read_one_h5_from_s3(bucket_name, s3_key)

    h5_file = h5py.File(path, 'r')
    print_h5_structure(h5_file)
    return h5_file

    
def _read_one_h5_from_s3(bucket_name, s3_key):
    """Download an HDF5 object from S3 into memory and open it with h5py."""
    try:
        s3_client = boto3.client('s3')

        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        
        file_content = response['Body'].read()
        buffer = io.BytesIO(file_content)
        
        h5_file = h5py.File(buffer, 'r')
        print_h5_structure(h5_file)
        return h5_file

    except Exception as e:
        mylogger.info(f"Error reading from S3: {e}")
        exit()

  
def get_hypertension_label(sbp, dbp):
    """
    Assigns a hypertension label based on SBP/DBP with an exclusion zone.

    Args:
        sbp (int or float): Systolic Blood Pressure.
        dbp (int or float): Diastolic Blood Pressure.

    Returns:
        int or None:
            - 1 for Hypertension
            - 0 for Normal
            - None for Excluded/Ambiguous cases
    """
    SBP_THRESHOLD = 130
    DBP_THRESHOLD = 80
    EXCLUSION_ZONE = 8

    sbp_upper_bound = SBP_THRESHOLD + EXCLUSION_ZONE  # 138
    sbp_lower_bound = SBP_THRESHOLD - EXCLUSION_ZONE  # 122
    
    dbp_upper_bound = DBP_THRESHOLD + EXCLUSION_ZONE  # 88
    dbp_lower_bound = DBP_THRESHOLD - EXCLUSION_ZONE  # 72

    if sbp >= sbp_upper_bound or dbp >= dbp_upper_bound:
        return 1  # Hypertension
        
    elif sbp <= sbp_lower_bound and dbp <= dbp_lower_bound:
        return 0  # Normal
        
    else:
        return None  # Excluded




def min_max_norm(signal):
    """Scale a signal to the range [-1, 1]."""
    min_val = signal.min()
    max_val = signal.max()
    return 2 * (signal - min_val) / (max_val - min_val) - 1

            

def _upsample(original_signal, fs_original=25, fs_new=100, duration=10):
    """Resample a signal from ``fs_original`` to ``fs_new`` unless it is already at the target length."""
    if len(original_signal) == fs_new * duration:
        return original_signal
    
    original_length = len(original_signal)
    duration_sec = original_length / fs_original
    
    new_length = int(original_length * fs_new / fs_original)
    
    upsampled_signal = signal.resample(original_signal, new_length)

    return upsampled_signal



def update_bp_labels(labels):
    """Map (id, SBP, DBP) rows to binary hypertension labels, dropping excluded rows."""
    keep_idx = []
    new_labels = []
    for i, (_, sbp, dbp) in enumerate(labels):
        out = get_hypertension_label(sbp, dbp)
        if out is not None:
            keep_idx.append(i)
            new_labels.append(out)

    return np.array(new_labels).reshape(-1, 1), keep_idx



def metrics_by_patient(test_patients, all_test_preds, all_test_probs, all_test_labels, args):
    """Compute accuracy, AUROC, and F1 per patient and return their means."""
    def get_average(d, key='acc'):
        return np.mean([v[key] for uid, v in d.items()])
        
    patient_preds = {}
    patient_probs = {}
    patient_labels = {}
    
    for i, patient_id in enumerate(test_patients):
        if patient_id not in patient_preds:
            patient_preds[patient_id] = []
            patient_probs[patient_id] = []
            patient_labels[patient_id] = []
        
        patient_preds[patient_id].append(all_test_preds[i])
        patient_probs[patient_id].append(all_test_probs[i])
        patient_labels[patient_id].append(all_test_labels[i])

    out = {}
    
    for pid in patient_preds:
        
        task = 'binary' if args.num_classes == 2 else 'multiclass'
        f1 = torchmetrics.F1Score(task=task, num_classes=args.num_classes)
        auroc = torchmetrics.AUROC(task=task, num_classes=args.num_classes)
        acc = torchmetrics.Accuracy(task=task, num_classes=args.num_classes)
        
        preds = torch.tensor(patient_preds[pid])
        probs = torch.tensor(patient_probs[pid])
        labels = torch.tensor(patient_labels[pid])
        nsamples = len(patient_preds[pid])
        
        f1.update(preds, labels)
        auroc.update(probs, labels)
        acc.update(preds, labels)

        out[pid] = {'acc'   : acc.compute().item(),
                    'auroc' : auroc.compute().item(),
                    'f1'    : f1.compute().item(),
                    'nsamples': nsamples,
                   }

    return get_average(out, 'acc'), get_average(out, 'auroc'), get_average(out, 'f1')
    


class ClassificationDataset(Dataset):
    """A dataset that takes pre-loaded data and indices for a specific split."""
    def __init__(self, ppg, labels, indices, args):
        super().__init__()
        self.ppg = ppg[indices]
        self.labels = labels[indices]
        self.app = args.app

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = min_max_norm(self.ppg[idx])
        signal = _upsample(signal, fs_original=TASK_DEF[self.app]['fs'])
        signal = torch.from_numpy(signal).float()
        label = torch.tensor(int(self.labels[idx][0]), dtype=torch.long) 
        
        return signal, label





def prepare_data(args, h5_buffer):
    """Load signals, patient IDs, and labels from the HDF5 file into memory."""
    all_ppg = h5_buffer['ppg'][:]
    all_patient_ids = h5_buffer['patient_ids'][:]
    all_labels = h5_buffer['labels'][:]
    
    if args.app == 'element':
        all_labels, keep_idx = update_bp_labels(all_labels)
        all_ppg = all_ppg[keep_idx]
        all_patient_ids = all_patient_ids[keep_idx]

    mylogger.info("Data loaded from H5 buffer into memory.")

    if args.to_summary:
        summarize_dataset(all_patient_ids, all_labels, args)

    return all_ppg, all_patient_ids, all_labels
                               


def run_linear_probe_random(args, h5_buffer):
    """Linear-probe k-fold CV with stratified segment-level splits."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mylogger.info(f"Using device: {device}")
    
    all_ppg, all_patient_ids, all_labels = prepare_data(args, h5_buffer)
    
    # Load backbone weights from the Lightning checkpoint.
    backbone = get_model(args.cfg)
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    lightning_state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict([(k[len('model.'):], v) for k, v in lightning_state_dict.items() if k.startswith('model.')])
    backbone.load_state_dict(new_state_dict)
    mylogger.info("Backbone loaded successfully.")

    
    sgkf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    
    fold_metrics = {'f1': [], 'auroc': [], 'acc': []}

    for fold, (train_val_idx, test_idx) in enumerate(sgkf.split(all_ppg, all_labels)):
        
        mylogger.info(f"\n{'='*20} FOLD {fold+1}/{args.k_folds} {'='*20}")

        # Hold out 1/8 of the training pool for validation.
        sgkf2 = StratifiedKFold(n_splits=8, shuffle=True, random_state=args.seed)
        
        train_val_ppg = all_ppg[train_val_idx]
        train_val_labels = all_labels[train_val_idx]
        train_val_patient_ids = all_patient_ids[train_val_idx]
        (train_idx, val_idx) = next(iter(sgkf2.split(train_val_ppg, train_val_labels)))

        mylogger.info(f'Train vs Val vs Test: {len(train_idx)}, {len(val_idx)}, {len(test_idx)}')
        mylogger.info('F1 vs AUROC vs ACC')
        
        train_dataset = ClassificationDataset(train_val_ppg, train_val_labels, train_idx, args)
        val_dataset = ClassificationDataset(train_val_ppg, train_val_labels, val_idx, args)
        test_dataset = ClassificationDataset(all_ppg, all_labels, test_idx, args) 
    
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size) 
        
        # Fresh probe and optimizer for each fold.
        probe_model = LinearProbeModel(backbone, args.num_classes).to(device)
        optimizer = torch.optim.Adam(probe_model.classifier.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        best_model_weights = None
        last_update = 0
        # Train with early stopping on validation loss.
        for epoch in range(args.epochs):
            probe_model.train()
            for data, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = probe_model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            probe_model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for data, labels in val_loader:
                    data, labels = data.to(device), labels.to(device)
                    outputs = probe_model(data)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_weights = copy.deepcopy(probe_model.state_dict())
                last_update = epoch

            if epoch - last_update > args.patience:
                break
        
        mylogger.info(f"Fold {fold+1} Best Val Loss: {best_val_loss:.4f}")

        # Evaluate the best checkpoint of this fold on the held-out test fold.
        probe_model.load_state_dict(best_model_weights)
        probe_model.eval()

        task = 'binary' if args.num_classes == 2 else 'multiclass'
        f1 = torchmetrics.F1Score(task=task, num_classes=args.num_classes).to(device)
        auroc = torchmetrics.AUROC(task=task, num_classes=args.num_classes).to(device)
        acc = torchmetrics.Accuracy(task=task, num_classes=args.num_classes).to(device)

        
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(device)
                outputs = probe_model(data)
                preds = torch.argmax(outputs, dim=1)
                probs = torch.softmax(outputs, dim=1)[:, 1]

                f1.update(preds, labels.to(device))
                auroc.update(probs, labels.to(device))
                acc.update(preds, labels.to(device))

        fold_metrics['f1'].append(f1.compute().item())
        fold_metrics['auroc'].append(auroc.compute().item())
        fold_metrics['acc'].append(acc.compute().item())
        
        mylogger.info(f"{fold_metrics['f1'][-1]:.4f} {fold_metrics['auroc'][-1]:.4f} {fold_metrics['acc'][-1]:.4f}")

    # Mean ± std across folds: F1, AUROC, accuracy.
    mylogger.info(f"\n{'='*20} K-Fold Cross-Validation Results ({args.k_folds} Folds) {'='*20}")
    mylogger.info(f"{np.mean(fold_metrics['f1']):.4f} ± {np.std(fold_metrics['f1']):.4f}")
    mylogger.info(f"{np.mean(fold_metrics['auroc']):.4f} ± {np.std(fold_metrics['auroc']):.4f}")
    mylogger.info(f"{np.mean(fold_metrics['acc']):.4f} ± {np.std(fold_metrics['acc']):.4f}")


            


def run_linear_probe_by_patient(args, h5_buffer):
    """Linear-probe k-fold CV with patient-level splits; metrics are averaged per patient."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mylogger.info(f"Using device: {device}")
    
    all_ppg, all_patient_ids, all_labels = prepare_data(args, h5_buffer)

    
    # Load backbone weights from the Lightning checkpoint.
    backbone = get_model(args.cfg)
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    lightning_state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict([(k[len('model.'):], v) for k, v in lightning_state_dict.items() if k.startswith('model.')])
    backbone.load_state_dict(new_state_dict)
    mylogger.info("Backbone loaded successfully.")


    # Split by patient so no participant appears in both train and test.
    sgkf = StratifiedGroupKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    
    fold_metrics = {'f1': [], 'auroc': [], 'per_patient_acc': [], 'acc': []}

    for fold, (train_val_idx, test_idx) in enumerate(sgkf.split(all_ppg, all_labels, groups=all_patient_ids)):
        
        mylogger.info(f"\n{'='*20} FOLD {fold+1}/{args.k_folds} {'='*20}")
        
        # Patient-level validation split within the training pool.
        sgkf2 = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=args.seed)
        
        train_val_ppg = all_ppg[train_val_idx]
        train_val_labels = all_labels[train_val_idx]
        train_val_patient_ids = all_patient_ids[train_val_idx]
        (train_idx, val_idx) = next(iter(sgkf2.split(train_val_ppg, 
                                                     train_val_labels, 
                                                     groups=train_val_patient_ids)))

        mylogger.info(f'Train vs Val vs Test: {len(train_idx)}, {len(val_idx)}, {len(test_idx)}')
        mylogger.info('F1 vs AUROC vs ACC')
        
        train_dataset = ClassificationDataset(train_val_ppg, train_val_labels, train_idx, args)
        val_dataset = ClassificationDataset(train_val_ppg, train_val_labels, val_idx, args)
        test_dataset = ClassificationDataset(all_ppg, all_labels, test_idx, args) 
    
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size) 
        
        # Fresh probe and optimizer for each fold.
        probe_model = LinearProbeModel(backbone, args.num_classes).to(device)
        optimizer = torch.optim.Adam(probe_model.classifier.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        best_model_weights = None

        last_update = 0
        # Train with early stopping on validation loss.
        for epoch in range(args.epochs):
            probe_model.train()
            for data, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = probe_model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            probe_model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for data, labels in val_loader:
                    data, labels = data.to(device), labels.to(device)
                    outputs = probe_model(data)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            if avg_val_loss < best_val_loss :
                best_val_loss = avg_val_loss
                best_model_weights = copy.deepcopy(probe_model.state_dict())
                last_update = epoch

            if epoch - last_update > args.patience:
                break
        
        mylogger.info(f"Fold {fold+1} Best Val Loss: {best_val_loss:.4f}")


        # Evaluate the best checkpoint of this fold on the held-out test fold.
        probe_model.load_state_dict(best_model_weights)
        probe_model.eval()

        all_test_preds = []
        all_test_probs = []
        all_test_labels = []
        
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(device)
                outputs = probe_model(data)
                preds = torch.argmax(outputs, dim=1)
                probs = torch.softmax(outputs, dim=1)[:, 1]  # probability of the positive class
                
                all_test_preds.extend(preds.cpu().numpy())
                all_test_probs.extend(probs.cpu().numpy())
                all_test_labels.extend(labels.cpu().numpy())
               
        acc, auroc, f1 = metrics_by_patient(all_patient_ids[test_idx], all_test_preds, all_test_probs, all_test_labels, args)
       
        fold_metrics['f1'].append(f1)
        fold_metrics['auroc'].append(auroc)
        fold_metrics['acc'].append(acc)
        
        mylogger.info(f"{fold_metrics['f1'][-1]:.4f} {fold_metrics['auroc'][-1]:.4f} {fold_metrics['acc'][-1]:.4f}")

    # Mean ± std across folds: F1, AUROC, accuracy.
    mylogger.info(f"\n{'='*20} By-Patient K-Fold Cross-Validation Results ({args.k_folds} Folds {'='*20}")
    mylogger.info(f"{np.mean(fold_metrics['f1']):.4f} ± {np.std(fold_metrics['f1']):.4f}")
    mylogger.info(f"{np.mean(fold_metrics['auroc']):.4f} ± {np.std(fold_metrics['auroc']):.4f}")
    mylogger.info(f"{np.mean(fold_metrics['acc']):.4f} ± {np.std(fold_metrics['acc']):.4f}")

if __name__ == '__main__':
    
 
    parser = argparse.ArgumentParser(
        description="Linear-probe evaluation of a pretrained backbone on a binary downstream task."
    )

    parser.add_argument(
        '--source', 
        type=str, 
        default='ppg+ecg',
        help="Input modality: 'ppg', 'ecg', or 'ppg+ecg'.",
    )
    
    parser.add_argument(
        '--model',
        type = str, 
        default='himae',
        help="Backbone: 'himae', 'himae_25m', 'himae_pulsedb', or a PulseBridge variant.",
    )
    parser.add_argument(
        '--app',
        type = str, 
        default='pvc',
        help="Downstream task key in TASK_DEF (e.g., 'pvc', 'element').",
    )
    
    parser.add_argument(
        '--split',
        type = str, 
        default='random',
        help="Cross-validation split: 'random' (segment-level) or 'patient'.",
    )
    parser.add_argument(
        '--batch',
        type = int, 
        default=2048,
        help='Batch size.',
    )
    parser.add_argument(
        '--epoch',
        type = int, 
        default=7,
        help='Maximum training epochs per fold.',
    )
    
    inputs = parser.parse_args()
    
    LinearProbeModel = HIMAE_ if 'himae' in inputs.model else PULSE_
    mylogger = create_logger("mylogger", f'logs/eval/{inputs.model}_{inputs.app}_{inputs.split}.log',
                             level=logging.DEBUG)
    
  
    class Args:
        
        if inputs.model == 'himae':
            ckpt_path = './checkpoints/REDACTED/ppg.ckpt'
        elif inputs.model == 'himae_25m':
            ckpt_path = './checkpoints/REDACTED/HiMAE-25M-segments.ckpt'
        elif inputs.model == 'himae_pulsedb':
            ckpt_path = './checkpoints/REDACTED/best-model.ckpt'
        num_classes = 2
        k_folds = 5
        seed = 77
        to_summary = False  # set True to write dataset summary files
        patience = 5
        learning_rate = 1e-3
        
        
        epochs     = inputs.epoch
        batch_size = inputs.batch
        app        = inputs.app
        by         = inputs.split

        cfg = {
            'model': inputs.model,
            'sampling_freq': 100,
            'seg_len': 10,   
            'source': inputs.source,
            'model_params': {
                  'latent_dim': 384,
                  'd_model': 512,              
                  'nhead': 8,                    
                  'depth_ecg': 3,                
                  'depth_ppg': 3,                
                  'depth_bridge': 2,             
                  'stem_ch': 32,                 
                  'dropout': 0.1,                
                  'use_cross_bridge': True,      
                  'use_fir_prior': True,         
                  'fir_kernel_size': 9,          
                  'fir_bandwidth': 2,           
                  'patch_len': 40,
            },
        }

    args = Args()

    h5_buffer = _read_one_h5_from_local(BUCKET, TASK_DEF.get(args.app, None)['h5_path'])

    if h5_buffer:
        if args.by == 'patient':
            run_linear_probe_by_patient(args, h5_buffer)
        else:
            run_linear_probe_random(args, h5_buffer)
            




