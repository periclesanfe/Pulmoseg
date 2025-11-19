"""
PulmoSeg Deep Learning - Script Principal de Treinamento
Treina modelos de segmentação de nódulos pulmonares
"""

import argparse
import torch
import torch.optim as optim

from src.config import cfg
from src.dataset import get_dataloaders
from src.augmentation import get_training_augmentation, get_validation_augmentation
from src.models.unet import get_model
from src.losses import get_loss_function
from src.trainer import Trainer


def parse_args():
    """Parse argumentos de linha de comando"""
    parser = argparse.ArgumentParser(
        description='PulmoSeg DL - Treinamento de Modelos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  # Treinar U-Net com configurações padrão
  python train.py

  # Treinar U-Net++ com encoder ResNet50
  python train.py --model unetplusplus --encoder resnet50

  # Treinar Attention U-Net por 100 epochs
  python train.py --model manet --epochs 100

  # Treinar sem pesos pré-treinados
  python train.py --encoder-weights None

  # Usar batch size maior
  python train.py --batch-size 16
        """
    )

    parser.add_argument('--model', type=str, default=cfg.MODEL_NAME,
                       help=f'Arquitetura do modelo (padrão: {cfg.MODEL_NAME})')
    parser.add_argument('--encoder', type=str, default=cfg.ENCODER_NAME,
                       help=f'Encoder pré-treinado (padrão: {cfg.ENCODER_NAME})')
    parser.add_argument('--encoder-weights', type=str, default=cfg.ENCODER_WEIGHTS,
                       help=f'Pesos do encoder (padrão: {cfg.ENCODER_WEIGHTS})')
    parser.add_argument('--epochs', type=int, default=cfg.NUM_EPOCHS,
                       help=f'Número de epochs (padrão: {cfg.NUM_EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=cfg.BATCH_SIZE,
                       help=f'Batch size (padrão: {cfg.BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=cfg.LEARNING_RATE,
                       help=f'Learning rate (padrão: {cfg.LEARNING_RATE})')
    parser.add_argument('--loss', type=str, default=cfg.LOSS_FUNCTION,
                       help=f'Loss function (padrão: {cfg.LOSS_FUNCTION})')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Desabilitar data augmentation')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Nome do experimento (padrão: model_encoder)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Atualizar configurações com argumentos
    cfg.MODEL_NAME = args.model
    cfg.ENCODER_NAME = args.encoder
    cfg.ENCODER_WEIGHTS = None if args.encoder_weights == 'None' else args.encoder_weights
    cfg.NUM_EPOCHS = args.epochs
    cfg.BATCH_SIZE = args.batch_size
    cfg.LEARNING_RATE = args.lr
    cfg.LOSS_FUNCTION = args.loss
    cfg.AUGMENTATION_ENABLED = not args.no_augmentation

    # Nome do experimento
    experiment_name = args.experiment_name or f"{cfg.MODEL_NAME}_{cfg.ENCODER_NAME}"

    # Print configurações
    print(cfg)

    # Device
    device = cfg.get_device()
    print(f"\nUsando device: {device}")

    # Criar DataLoaders
    print("\n" + "="*70)
    print("Preparando Dataset...")
    print("="*70)

    train_transform = get_training_augmentation() if cfg.AUGMENTATION_ENABLED else get_validation_augmentation()
    val_transform = get_validation_augmentation()

    train_loader, val_loader, test_loader, stats = get_dataloaders(
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS
    )

    # Criar modelo
    print("\n" + "="*70)
    print("Criando Modelo...")
    print("="*70)

    model = get_model(
        model_name=cfg.MODEL_NAME,
        encoder_name=cfg.ENCODER_NAME,
        encoder_weights=cfg.ENCODER_WEIGHTS
    )

    model.summary()

    # Loss function
    criterion = get_loss_function(
        loss_name=cfg.LOSS_FUNCTION,
        dice_weight=cfg.DICE_WEIGHT,
        bce_weight=cfg.BCE_WEIGHT
    )

    print(f"Loss function: {cfg.LOSS_FUNCTION}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=cfg.LR_FACTOR,
        patience=cfg.LR_PATIENCE,
        min_lr=cfg.LR_MIN,
        verbose=True
    )

    # Criar Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        experiment_name=experiment_name
    )

    # Treinar
    best_dice = trainer.train(num_epochs=cfg.NUM_EPOCHS)

    print(f"\n✓ Treinamento concluído!")
    print(f"✓ Melhor Dice Score (validação): {best_dice:.4f}")
    print(f"✓ Checkpoint salvo em: {trainer.checkpoint_dir / 'best.pth'}")


if __name__ == '__main__':
    main()
