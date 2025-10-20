import argparse
import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

from models import get_model_from_config 
from data import SpecimenSpecificDataset, LOSODataset, CombinedDataset

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
parser.add_argument('--model_type', type=str, default=None,
                    help='Model type to use (overrides config). Options: CNNCatCross')
parser.add_argument('--data_mode', type=str, default='specimen_specific', choices=['specimen_specific', 'loso_kfold'],
                    help='Data mode: "specimen_specific" or "loso_kfold"')
parser.add_argument('--fold', type=str, default='fold1', help='Fold to use for LOSO k-fold cross validation')
parser.add_argument('--dry_run', action='store_true', help='Create output directories and exit without training')
args = parser.parse_args()

def main():
    # Load configuration from YAML
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    data_mode = args.data_mode
    
    # Use model_type to organize output folders.
    model_folder = args.model_type if args.model_type is not None else "default"
    if args.data_mode == "loso_kfold":
        checkpoint_dir = os.path.join(config['checkpoint']['dir'], model_folder, args.fold)
        loss_dir = os.path.join(config['output']['loss_dir'], model_folder, args.fold)
        model_dir = os.path.join(config['output']['model_dir'], model_folder, args.fold)
    else:
        checkpoint_dir = os.path.join(config['checkpoint']['dir'], model_folder)
        loss_dir = os.path.join(config['output']['loss_dir'], model_folder)
        model_dir = os.path.join(config['output']['model_dir'], model_folder)

    # Create directories if they don't exist.
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # If dry run flag is set, just print the directories and exit.
    if args.dry_run:
        logging.info("Dry run mode: The following directories have been created/verified:")
        logging.info(f"Checkpoint Directory: {checkpoint_dir}")
        logging.info(f"Loss Curve Directory: {loss_dir}")
        logging.info(f"Model Directory: {model_dir}")
        return
    
    # Continue with loading data and training...
    if data_mode == "specimen_specific":
        data_config = config['data_configs']['specimen_specific']
        real_dataset = SpecimenSpecificDataset(
            data_root=data_config['data_root_real'],
            spec_id=data_config['spec_id'],
            image_side_size=data_config['image_side_size'],
            label_csv=data_config['label_csv_real'], 
            is_real=True
        )
        simulated_dataset = SpecimenSpecificDataset(
            data_root=data_config['data_root_simulated'],
            spec_id=data_config['spec_id'],
            image_side_size=data_config['image_side_size'],
            label_csv=data_config['label_csv_simulated'],
            is_real=False
        )
        combined_dataset = CombinedDataset(real_dataset, simulated_dataset)
        train_size = int(0.8 * len(combined_dataset))
        val_size = len(combined_dataset) - train_size
        train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])
    
    elif data_mode == "loso_kfold":
        data_config = config['data_configs']['loso_kfold']
        fold = data_config['folds'][args.fold]
        real_train_dataset = LOSODataset(
            data_root=data_config['data_root_real'],
            selected_spec_ids=fold['train_ids'],
            image_side_size=data_config['image_side_size'],
            label_csv=data_config['label_csv_real'], 
            augmentation_factor=fold['augmentation_factor'], 
            is_real=True 
        )
        simulated_train_dataset = LOSODataset(
            data_root=data_config['data_root_simulated'],
            selected_spec_ids=fold['train_ids'],
            image_side_size=data_config['image_side_size'],
            label_csv=data_config['label_csv_simulated'],
            augmentation_factor=fold['augmentation_factor'], 
            is_real=False
        )
        real_val_dataset = LOSODataset(
            data_root=data_config['data_root_real'],
            selected_spec_ids=fold['val_ids'],
            image_side_size=data_config['image_side_size'],
            label_csv=data_config['label_csv_real'],
            augmentation_factor=0,
            is_real=True 
        )
        simulated_val_dataset = LOSODataset(
            data_root=data_config['data_root_simulated'],
            selected_spec_ids=fold['val_ids'],
            image_side_size=data_config['image_side_size'],
            label_csv=data_config['label_csv_simulated'],
            augmentation_factor=0,
            is_real=False 
        )
        train_dataset = CombinedDataset(real_train_dataset, simulated_train_dataset)
        val_dataset = CombinedDataset(real_val_dataset, simulated_val_dataset)
    else:
        raise ValueError(f"Unsupported data_mode: {data_mode}")
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model_from_config(args.model_type, config).to(device)
    logging.info(f"Model device: {next(model.parameters()).device}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'], weight_decay=config['training'].get('wd', 0))
    
    num_epochs = config['training']['num_epochs']
    best_val_loss = float('inf')
    best_epoch = -1
    patience = config['training'].get('patience', 5)
    epochs_without_improvement = 0
    grad_scaler = torch.GradScaler("cuda")
    
    def train_epoch(model, dataloader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        for xray, drr, labels in tqdm(dataloader, desc="Training", leave=False):
            xray, drr = xray.to(device), drr.to(device)
            labels = labels.to(device).unsqueeze(1).float()
            optimizer.zero_grad()
            with torch.autocast("cuda"):
                outputs = model(xray, drr)
                loss = criterion(outputs, labels)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            running_loss += loss.item() * xray.size(0)
        return running_loss / len(dataloader.dataset)
    
    def evaluate_epoch(model, dataloader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xray, drr, labels in tqdm(dataloader, desc="Evaluating", leave=False):
                xray, drr = xray.to(device), drr.to(device)
                labels = labels.to(device).unsqueeze(1).float()
                with torch.autocast("cuda"):
                    outputs = model(xray, drr)
                    loss = criterion(outputs, labels)
                running_loss += loss.item() * xray.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                correct += (preds.float() == labels).sum().item()
                total += labels.size(0)
        return running_loss / total, correct / total
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_acc = evaluate_epoch(model, train_loader, criterion, device)[1]
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch+1
            epochs_without_improvement = 0
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_checkpoint_path)
            logging.info(f"--> Best model updated at epoch {epoch+1}")
        else:
            epochs_without_improvement += 1
            logging.info(f"No improvement for {epochs_without_improvement} epoch(s).")
        
        if epochs_without_improvement >= patience:
            logging.info(f"Early stopping triggered at epoch {epoch+1} after {patience} epochs without improvement.")
            break
    
    logging.info(f"Training complete. Best model at epoch {best_epoch} with val loss: {best_val_loss:.4f}")
    
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")
    
    plt.subplot(1,2,2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curves")
    
    plt.tight_layout()
    loss_curve_path = os.path.join(loss_dir, 'training_curves.png')
    plt.savefig(loss_curve_path)
    plt.close()
    
    model_path = os.path.join(model_dir, 'registration_model_weights.pth')
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model weights saved to {model_path}")

if __name__ == '__main__':
    main()