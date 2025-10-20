import argparse
import yaml
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models.factory import get_model_from_config
from data.dataset import SpecimenSpecificDataset, LOSODataset, CombinedDataset
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
parser.add_argument('--model_type', type=str, default=None,
                    help='Model type to use (overrides config). Options: CNNCatCross')
parser.add_argument('--data_mode', type=str, default='specimen_specific', choices=['specimen_specific', 'loso_kfold'],
                    help='Data mode: "specimen_specific" or "loso_kfold"')
args = parser.parse_args()

def objective(trial, config, device):
    # Override some training hyperparameters using trial suggestions
    training_config = config['training']
    hpo_config = config['hpo']
    
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    
    training_config['lr'] = lr
    training_config['wd'] = wd
    training_config['batch_size'] = batch_size

    data_mode = args.data_mode

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
        train_dataset, test_dataset = random_split(combined_dataset, [train_size, val_size])
    
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
        test_dataset = CombinedDataset(real_val_dataset, simulated_val_dataset)
    else:
        raise ValueError(f"Unsupported data_mode: {data_mode}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Determine model type: use command-line argument if provided; otherwise use config
    model_type = args.model_type if args.model_type is not None else config['model'].get('type', 'CNNCatCross')
    model = get_model_from_config(model_type, config).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    # Train for a short number of epochs (from HPO config)
    epochs = hpo_config.get('epochs_short', 3)
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        running_loss = 0.0
        for xray, drr, labels in tqdm(train_loader, desc="Training", leave=False):
            xray, drr = xray.to(device), drr.to(device)
            labels = labels.to(device).unsqueeze(1).float()
            optimizer.zero_grad()
            outputs = model(xray, drr)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xray.size(0)
    # Evaluate on the test dataset after training
    model.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for xray, drr, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            xray, drr = xray.to(device), drr.to(device)
            labels = labels.to(device).unsqueeze(1).float()
            outputs = model(xray, drr)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * xray.size(0)
            total += xray.size(0)
    avg_loss = total_loss / total
    return avg_loss

def main():
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, config, device), n_trials=config['hpo']['n_trials'])
    optuna.visualization.plot_optimization_history(study)
    
    print("Best hyperparameters:", study.best_trial.params)

if __name__ == '__main__':
    main()