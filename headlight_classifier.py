#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Headlight-Based Car Brand/Model Classifier - Optimized Version

Bu sistem, far görüntülerinden araç marka ve modelini tespit eden
derin öğrenme modeli eğitir ve kullanır.

Model Mimarisi:
- Base: EfficientNet-B0 (transfer learning)
- Custom Head: Optimized FC layers with batch normalization
- Optimization: AdamW with CosineAnnealingLR
- Augmentation: Moderate image augmentations
- Regularization: Dropout, Label Smoothing, Weight Decay

Kullanım:
    1. Model eğit: python headlight_classifier.py --train
    2. Tespit yap: python headlight_classifier.py --predict headlight.jpg
    3. Değerlendir: python headlight_classifier.py --evaluate
"""

import os
import sys
import json
import time
import argparse
import random
from pathlib import Path
from collections import defaultdict, Counter
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torchvision import transforms, models
import timm

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Reproducibility
def set_seed(seed=42):
    """Rastgelelik seed'ini ayarla"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


class HeadlightDataset(Dataset):
    """Far görüntüleri dataset sınıfı"""

    def __init__(self, df, transform=None, base_path="", is_training=True):
        """
        Args:
            df: Pandas DataFrame
            transform: Torchvision transforms
            base_path: Base path for images
            is_training: Training mode flag
        """
        self.df = df.copy().reset_index(drop=True)
        self.transform = transform
        self.base_path = Path(base_path)
        self.is_training = is_training

        # Class mapping - use only classes that appear in this dataset
        self.classes = sorted(df['stanford_class_id'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # Class names mapping
        self.class_names = dict(zip(df['stanford_class_id'], df['stanford_class_name']))

        logger.info(f"Dataset initialized with {len(self.df)} samples and {len(self.classes)} classes")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image path
        img_path = self.base_path / row['headlight_path']

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Could not load image {img_path}: {e}")
            # Create dummy image
            image = Image.new('RGB', (224, 224), color='black')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get label
        class_id = row['stanford_class_id']
        label = self.class_to_idx[class_id]

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'class_id': class_id,
            'class_name': row['stanford_class_name'],
            'image_path': str(img_path)
        }


class ImprovedHeadlightClassifier(nn.Module):
    """Improved headlight classifier with better architecture"""

    def __init__(self, num_classes, model_name='efficientnet_b0', dropout_rate=0.2,
                 freeze_base=True, hidden_size=512):
        super().__init__()

        self.num_classes = num_classes
        self.model_name = model_name

        # Load pretrained model
        if model_name == 'efficientnet_b0':
            self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
            num_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()
        elif model_name == 'efficientnet_b1':
            self.base_model = timm.create_model('efficientnet_b1', pretrained=True)
            num_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()
        elif model_name == 'resnet34':
            self.base_model = models.resnet34(pretrained=True)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Freeze base model layers initially
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Custom classifier with batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_size // 2, num_classes)
        )

        # Initialize classifier weights
        self._init_classifier_weights()

        logger.info(f"Model initialized: {model_name}, features: {num_features}, "
                    f"hidden: {hidden_size}, classes: {num_classes}")

    def _init_classifier_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.base_model(x)
        output = self.classifier(features)
        return output

    def unfreeze_base_layers(self, num_layers=10):
        """Gradually unfreeze base model layers"""
        # Get all parameters as a list
        base_params = list(self.base_model.parameters())

        # Unfreeze last num_layers
        for param in base_params[-num_layers:]:
            param.requires_grad = True

        logger.info(f"Unfroze last {num_layers} layers of base model")


class HeadlightTrainer:
    """Optimized model trainer class"""

    def __init__(self, output_dir="headlight_classifier_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optimized training config
        self.config = {
            'model_name': 'efficientnet_b0',
            'batch_size': 32,
            'num_epochs': 50,
            'learning_rate': 3e-4,  # Higher initial LR
            'weight_decay': 1e-4,
            'dropout_rate': 0.2,  # Lower dropout
            'image_size': 224,
            'patience': 15,
            'min_lr': 1e-6,
            'label_smoothing': 0.1,
            'hidden_size': 512,
            'freeze_epochs': 5,  # Freeze base model for first N epochs
            'gradient_clip': 1.0,
            'warmup_epochs': 3
        }

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Metrics storage
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_top5_accs = []
        self.val_top5_accs = []

    def load_data(self):
        """Load and prepare datasets"""
        logger.info("Loading datasets...")

        # Load CSVs
        train_csv = "enhanced_headlight_dataset/train/train_headlight_mapping.csv"
        test_csv = "enhanced_headlight_dataset/test/test_headlight_mapping.csv"

        if not os.path.exists(train_csv) or not os.path.exists(test_csv):
            raise FileNotFoundError("CSV files not found!")

        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)

        # Add split column to test_df if not exists
        if 'split' not in test_df.columns:
            test_df['split'] = 'test'

        logger.info(f"Train samples: {len(train_df)}")
        logger.info(f"Test samples: {len(test_df)}")

        # Dataset analysis
        self._analyze_dataset(train_df, test_df)

        return train_df, test_df

    def _analyze_dataset(self, train_df, test_df):
        """Analyze dataset distribution"""
        logger.info("\n=== DATASET ANALYSIS ===")

        # Combine for full analysis
        full_df = pd.concat([train_df, test_df], ignore_index=True)

        # Class distribution
        class_counts = full_df['stanford_class_name'].value_counts()
        logger.info(f"Number of unique classes: {len(class_counts)}")
        logger.info(f"Min samples per class: {class_counts.min()}")
        logger.info(f"Max samples per class: {class_counts.max()}")
        logger.info(f"Avg samples per class: {class_counts.mean():.1f}")

        # Filter classes with too few samples
        min_samples_per_class = 5
        valid_classes = class_counts[class_counts >= min_samples_per_class].index
        logger.info(f"Classes with >= {min_samples_per_class} samples: {len(valid_classes)}")

    def get_transforms(self):
        """Get optimized data augmentation transforms"""
        # Moderate augmentation for training
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to larger size first
            transforms.RandomCrop(self.config['image_size']),  # Random crop
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),  # Reduced rotation
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # No augmentation for validation
        val_transform = transforms.Compose([
            transforms.Resize((self.config['image_size'], self.config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return train_transform, val_transform

    def create_dataloaders(self, train_df, val_df, test_df):
        """Create data loaders with balanced sampling"""
        train_transform, val_transform = self.get_transforms()

        # Filter classes with too few samples
        min_samples = 5
        class_counts = train_df['stanford_class_id'].value_counts()
        valid_classes = class_counts[class_counts >= min_samples].index.tolist()

        # Filter dataframes
        train_df_filtered = train_df[train_df['stanford_class_id'].isin(valid_classes)].copy()
        val_df_filtered = val_df[val_df['stanford_class_id'].isin(valid_classes)].copy()
        test_df_filtered = test_df[test_df['stanford_class_id'].isin(valid_classes)].copy()

        logger.info(f"After filtering: Train {len(train_df_filtered)}, "
                    f"Val {len(val_df_filtered)}, Test {len(test_df_filtered)}")

        # Create datasets
        train_dataset = HeadlightDataset(train_df_filtered, transform=train_transform, is_training=True)
        val_dataset = HeadlightDataset(val_df_filtered, transform=val_transform, is_training=False)
        test_dataset = HeadlightDataset(test_df_filtered, transform=val_transform, is_training=False)

        # Ensure all datasets have same classes
        all_classes = set()
        all_classes.update(train_dataset.classes)
        all_classes.update(val_dataset.classes)
        all_classes.update(test_dataset.classes)
        num_classes = len(all_classes)

        # Create balanced sampler for training
        class_counts = Counter([row['stanford_class_id'] for _, row in train_dataset.df.iterrows()])
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[row['stanford_class_id']] for _, row in train_dataset.df.iterrows()]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            sampler=sampler,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False,
            drop_last=True  # Drop last incomplete batch
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        return train_loader, val_loader, test_loader, num_classes

    def train_epoch(self, model, train_loader, criterion, optimizer, scheduler, epoch):
        """Train one epoch with gradient clipping"""
        model.train()
        running_loss = 0.0
        correct = 0
        correct_top5 = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['gradient_clip'])

            optimizer.step()

            # Update learning rate (for warmup)
            if epoch <= self.config['warmup_epochs']:
                scheduler.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, 1, True, True)
            correct_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss / (batch_idx + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%',
                'top5': f'{100. * correct_top5 / total:.2f}%'
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        epoch_top5_acc = 100. * correct_top5 / total

        return epoch_loss, epoch_acc, epoch_top5_acc

    def validate(self, model, val_loader, criterion):
        """Validate model"""
        model.eval()
        running_loss = 0.0
        correct = 0
        correct_top5 = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Top-5 accuracy
                _, top5_pred = outputs.topk(5, 1, True, True)
                correct_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        epoch_top5_acc = 100. * correct_top5 / total

        return epoch_loss, epoch_acc, epoch_top5_acc

    def train(self):
        """Main training function with optimizations"""
        logger.info("Starting optimized training...")

        # Load data
        train_df, test_df = self.load_data()

        # Filter classes with at least min_samples_per_class samples
        min_samples_per_class = 2  # Minimum 2 sample per class for stratification
        class_counts = train_df['stanford_class_id'].value_counts()
        valid_classes = class_counts[class_counts >= min_samples_per_class].index.tolist()

        # Filter dataframes
        train_df_filtered = train_df[train_df['stanford_class_id'].isin(valid_classes)].copy()
        test_df_filtered = test_df[test_df['stanford_class_id'].isin(valid_classes)].copy()

        logger.info(f"After filtering - Train samples: {len(train_df_filtered)}, Test samples: {len(test_df_filtered)}")
        logger.info(f"Remaining classes: {len(valid_classes)}")

        # Create validation split - only stratify if we have enough samples
        try:
            train_df_split, val_df = train_test_split(
                train_df_filtered,
                test_size=0.2,
                random_state=42,
                stratify=train_df_filtered['stanford_class_id']
            )
            logger.info("Using stratified split for validation")
        except ValueError as e:
            logger.warning(f"Stratified split failed: {e}. Using random split instead")
            train_df_split, val_df = train_test_split(
                train_df_filtered,
                test_size=0.2,
                random_state=42
            )

        logger.info(f"Split sizes - Train: {len(train_df_split)}, Val: {len(val_df)}, Test: {len(test_df_filtered)}")

        # Create dataloaders
        train_loader, val_loader, test_loader, num_classes = self.create_dataloaders(
            train_df_split, val_df, test_df_filtered
        )

        # Rest of the training code remains the same...
        # Create model
        model = ImprovedHeadlightClassifier(
            num_classes=num_classes,
            model_name=self.config['model_name'],
            dropout_rate=self.config['dropout_rate'],
            freeze_base=True,  # Start with frozen base
            hidden_size=self.config['hidden_size']
        ).to(self.device)

        # Loss with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config['label_smoothing'])

        # Optimizer - different learning rates for base and classifier
        classifier_params = model.classifier.parameters()
        base_params = model.base_model.parameters()

        optimizer = optim.AdamW([
            {'params': base_params, 'lr': self.config['learning_rate'] * 0.1},
            {'params': classifier_params, 'lr': self.config['learning_rate']}
        ], weight_decay=self.config['weight_decay'])

        # Schedulers
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=self.config['warmup_epochs'] * len(train_loader)
        )

        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['num_epochs'] - self.config['warmup_epochs'],
            eta_min=self.config['min_lr']
        )

        # Training loop
        best_val_acc = 0.0
        best_val_top5_acc = 0.0
        patience_counter = 0

        for epoch in range(1, self.config['num_epochs'] + 1):
            logger.info(f"\nEpoch {epoch}/{self.config['num_epochs']}")

            # Unfreeze base model after freeze_epochs
            if epoch == self.config['freeze_epochs'] + 1:
                model.unfreeze_base_layers(num_layers=20)
                logger.info("Unfroze base model layers for fine-tuning")

            # Train
            if epoch <= self.config['warmup_epochs']:
                train_loss, train_acc, train_top5 = self.train_epoch(
                    model, train_loader, criterion, optimizer, warmup_scheduler, epoch
                )
            else:
                train_loss, train_acc, train_top5 = self.train_epoch(
                    model, train_loader, criterion, optimizer, None, epoch
                )

            # Validate
            val_loss, val_acc, val_top5 = self.validate(model, val_loader, criterion)

            # Adjust learning rate
            if epoch > self.config['warmup_epochs']:
                main_scheduler.step()

            # Log metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            self.train_top5_accs.append(train_top5)
            self.val_top5_accs.append(val_top5)

            current_lr = optimizer.param_groups[1]['lr']  # Classifier LR
            logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Top5: {train_top5:.2f}%")
            logger.info(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, Top5: {val_top5:.2f}%")
            logger.info(f"Learning Rate: {current_lr:.2e}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_top5_acc = val_top5
                patience_counter = 0

                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_top5_acc': val_top5,
                    'config': self.config,
                    'num_classes': num_classes,
                    'class_to_idx': train_loader.dataset.class_to_idx,
                    'idx_to_class': train_loader.dataset.idx_to_class,
                    'class_names': train_loader.dataset.class_names
                }

                torch.save(checkpoint, self.output_dir / 'best_model.pth')
                logger.info(f"✅ New best model saved! Val Acc: {val_acc:.2f}%, Top5: {val_top5:.2f}%")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.config['patience']:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

        # Plot training history
        self.plot_training_history()

        # Final evaluation on test set
        logger.info("\n=== Final Test Set Evaluation ===")
        test_loss, test_acc, test_top5 = self.validate(model, test_loader, criterion)
        logger.info(f"Test - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%, Top5: {test_top5:.2f}%")
        logger.info(f"Best Val Acc: {best_val_acc:.2f}%, Best Val Top5: {best_val_top5_acc:.2f}%")

        return model

    def plot_training_history(self):
        """Plot training curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(self.train_losses) + 1)

        # Loss plot
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Top-1 Accuracy plot
        ax2.plot(epochs, self.train_accs, 'b-', label='Train Acc')
        ax2.plot(epochs, self.val_accs, 'r-', label='Val Acc')
        ax2.set_title('Model Accuracy (Top-1)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)

        # Top-5 Accuracy plot
        ax3.plot(epochs, self.train_top5_accs, 'b-', label='Train Top-5')
        ax3.plot(epochs, self.val_top5_accs, 'r-', label='Val Top-5')
        ax3.set_title('Model Accuracy (Top-5)')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.legend()
        ax3.grid(True)

        # Learning curves comparison
        ax4.plot(epochs, np.array(self.train_accs) - np.array(self.val_accs), 'g-', label='Overfit Gap')
        ax4.set_title('Overfitting Analysis')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Train-Val Accuracy Gap (%)')
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=300)
        plt.close()

        logger.info(f"Training history saved to {self.output_dir / 'training_history.png'}")


class HeadlightPredictor:
    """Predictor class for inference"""

    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        self.num_classes = checkpoint['num_classes']
        self.class_to_idx = checkpoint['class_to_idx']
        self.idx_to_class = checkpoint['idx_to_class']
        self.class_names = checkpoint.get('class_names', {})

        # Create model
        self.model = ImprovedHeadlightClassifier(
            num_classes=self.num_classes,
            model_name=self.config['model_name'],
            dropout_rate=self.config.get('dropout_rate', 0.2),
            freeze_base=False,
            hidden_size=self.config.get('hidden_size', 512)
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((self.config['image_size'], self.config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Val Acc: {checkpoint.get('val_acc', 0):.2f}%, "
                    f"Val Top5: {checkpoint.get('val_top5_acc', 0):.2f}%")

    def predict(self, image_path, top_k=5):
        """Predict class for a single image"""
        # Load image
        image = Image.open(image_path).convert('RGB')

        # Transform
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1)

            # Get top-k predictions
            top_probs, top_indices = torch.topk(probs, min(top_k, self.num_classes))

        # Convert to lists
        top_probs = top_probs[0].cpu().numpy()
        top_indices = top_indices[0].cpu().numpy()

        # Prepare results
        results = []
        for prob, idx in zip(top_probs, top_indices):
            class_id = self.idx_to_class[idx]
            class_name = self.class_names.get(class_id, "Unknown")
            results.append({
                'class_id': class_id,
                'class_name': class_name,
                'probability': float(prob)
            })

        return results

    def predict_batch(self, image_paths, top_k=5):
        """Predict classes for a batch of images"""
        # Load and transform images
        images = []
        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
                images.append(image)
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
                continue

        if not images:
            return []

        # Stack images into batch
        input_tensor = torch.stack(images).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1)

            # Get top-k predictions for each image
            top_probs, top_indices = torch.topk(probs, min(top_k, self.num_classes), dim=1)

        # Convert to lists and prepare results
        batch_results = []
        for i in range(len(images)):
            image_results = []
            for prob, idx in zip(top_probs[i], top_indices[i]):
                class_id = self.idx_to_class[idx.item()]
                class_name = self.class_names.get(class_id, "Unknown")
                image_results.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'probability': prob.item()
                })
            batch_results.append({
                'image_path': image_paths[i],
                'predictions': image_results
            })

        return batch_results


def evaluate_model(test_loader, model, device, output_dir):
    """Evaluate model on test set with detailed metrics"""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    # Concatenate probabilities
    all_probs = np.concatenate(all_probs, axis=0)

    # Classification report
    class_names = [test_loader.dataset.idx_to_class[i] for i in range(len(test_loader.dataset.idx_to_class))]
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True
    )

    # Save report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_dir / 'classification_report.csv', index=True)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Top-k accuracy
    top3_acc = top_k_accuracy_score(all_labels, all_probs, k=3)
    top5_acc = top_k_accuracy_score(all_labels, all_probs, k=5)

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'top3_accuracy': top3_acc,
        'top5_accuracy': top5_acc,
        'num_classes': len(class_names),
        'num_samples': len(all_labels)
    }

    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Evaluation complete. Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Top-3 Accuracy: {top3_acc:.4f}, Top-5 Accuracy: {top5_acc:.4f}")

    return metrics


def main():
    """Main function for CLI"""
    parser = argparse.ArgumentParser(description='Headlight-Based Car Brand/Model Classifier')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', type=str, help='Path to image for prediction')
    parser.add_argument('--batch-predict', type=str, help='Path to folder with images for batch prediction')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model on test set')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='headlight_classifier_output', help='Output directory')

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.train:
        # Training mode
        trainer = HeadlightTrainer(output_dir=output_dir)
        trainer.train()

    elif args.predict:
        # Single prediction mode
        if not os.path.exists(args.predict):
            logger.error(f"Image file not found: {args.predict}")
            return

        predictor = HeadlightPredictor(output_dir / args.model)
        results = predictor.predict(args.predict)

        print("\nPrediction Results:")
        for i, res in enumerate(results, 1):
            print(f"{i}. {res['class_name']} (ID: {res['class_id']}): {res['probability']:.4f}")

    elif args.batch_predict:
        # Batch prediction mode
        if not os.path.isdir(args.batch_predict):
            logger.error(f"Directory not found: {args.batch_predict}")
            return

        image_paths = [os.path.join(args.batch_predict, f)
                       for f in os.listdir(args.batch_predict)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_paths:
            logger.error("No valid images found in directory")
            return

        predictor = HeadlightPredictor(output_dir / args.model)
        batch_results = predictor.predict_batch(image_paths)

        # Save results
        with open(output_dir / 'batch_predictions.json', 'w') as f:
            json.dump(batch_results, f, indent=2)

        print(f"\nBatch predictions saved to {output_dir / 'batch_predictions.json'}")

    elif args.evaluate:
        # Evaluation mode
        if not os.path.exists(output_dir / args.model):
            logger.error(f"Model file not found: {output_dir / args.model}")
            return

        # Load data
        trainer = HeadlightTrainer(output_dir=output_dir)
        train_df, test_df = trainer.load_data()

        # Create test dataset
        _, val_transform = trainer.get_transforms()
        test_dataset = HeadlightDataset(test_df, transform=val_transform, is_training=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=trainer.config['batch_size'],
            shuffle=False,
            num_workers=0
        )

        # Load model
        predictor = HeadlightPredictor(output_dir / args.model)

        # Evaluate
        metrics = evaluate_model(test_loader, predictor.model, trainer.device, output_dir)

        print("\nEvaluation Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Top-3 Accuracy: {metrics['top3_accuracy']:.4f}")
        print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
        print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()