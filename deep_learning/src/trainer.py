"""
PulmoSeg Deep Learning - Trainer
Training loop com early stopping, checkpointing e TensorBoard logging
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import time

from src.metrics import MetricsTracker
from src.config import cfg


class Trainer:
    """Classe para gerenciar treinamento de modelos"""

    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 scheduler=None, device=None, experiment_name='experiment'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or cfg.get_device()
        self.experiment_name = experiment_name

        # Move model to device
        self.model = self.model.to(self.device)

        # TensorBoard
        self.writer = SummaryWriter(cfg.TENSORBOARD_DIR / experiment_name)

        # Early stopping
        self.best_val_loss = float('inf')
        self.best_val_dice = 0.0
        self.epochs_without_improvement = 0

        # Checkpoints
        self.checkpoint_dir = cfg.CHECKPOINTS_DIR / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Mixed precision (para MPS/CUDA)
        self.use_amp = cfg.USE_MIXED_PRECISION and self.device.type in ['cuda', 'mps']
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None

    def train_epoch(self, epoch):
        """Treina uma época"""
        self.model.train()
        metrics = MetricsTracker(metrics_list=cfg.METRICS)
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')

        for batch_idx, (images, masks, _) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            if self.use_amp and self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp and self.scaler:
                self.scaler.scale(loss).backward()
                if cfg.GRADIENT_CLIP_VAL > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.GRADIENT_CLIP_VAL)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if cfg.GRADIENT_CLIP_VAL > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.GRADIENT_CLIP_VAL)
                self.optimizer.step()

            # Métricas
            metrics.update(outputs, masks)
            total_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(self.train_loader)
        avg_metrics = metrics.get_averages()

        return avg_loss, avg_metrics

    @torch.no_grad()
    def validate_epoch(self, epoch):
        """Valida uma época"""
        self.model.eval()
        metrics = MetricsTracker(metrics_list=cfg.METRICS)
        total_loss = 0.0

        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')

        for images, masks, _ in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            metrics.update(outputs, masks)
            total_loss += loss.item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = metrics.get_averages()

        return avg_loss, avg_metrics

    def save_checkpoint(self, epoch, val_loss, val_dice, is_best=False):
        """Salva checkpoint do modelo"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_dice': val_dice,
            'config': cfg.to_dict()
        }

        # Salvar último checkpoint
        last_path = self.checkpoint_dir / 'last.pth'
        torch.save(checkpoint, last_path)

        # Salvar melhor checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved (Dice: {val_dice:.4f})")

    def train(self, num_epochs):
        """Training loop completo"""
        print(f"\n{'='*70}")
        print(f"Iniciando Treinamento: {self.experiment_name}")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {cfg.BATCH_SIZE}")
        print(f"Learning rate: {cfg.LEARNING_RATE}")
        print(f"{'='*70}\n")

        start_time = time.time()

        for epoch in range(num_epochs):
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)

            # Validate
            val_loss, val_metrics = self.validate_epoch(epoch)

            # Learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            for metric_name, value in train_metrics.items():
                self.writer.add_scalar(f'Train/{metric_name}', value, epoch)
            for metric_name, value in val_metrics.items():
                self.writer.add_scalar(f'Val/{metric_name}', value, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Dice: {train_metrics['dice']:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Dice:   {val_metrics['dice']:.4f}")

            # Early stopping check
            is_best = val_metrics['dice'] > self.best_val_dice

            if is_best:
                self.best_val_dice = val_metrics['dice']
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, val_metrics['dice'], is_best=is_best)

            # Early stopping
            if self.epochs_without_improvement >= cfg.EARLY_STOPPING_PATIENCE:
                print(f"\n⚠ Early stopping triggered! No improvement for {cfg.EARLY_STOPPING_PATIENCE} epochs.")
                break

        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Treinamento Concluído!")
        print(f"{'='*70}")
        print(f"Total time: {total_time/60:.2f} min")
        print(f"Best Val Dice: {self.best_val_dice:.4f}")
        print(f"Best model saved at: {self.checkpoint_dir / 'best.pth'}")
        print(f"{'='*70}\n")

        self.writer.close()

        return self.best_val_dice
