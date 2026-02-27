import logging
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
import yaml
from models import get_model_from_config
from data.dataset import SpecimenSpecificDataset, CombinedDataset

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=None,
                    help='Path to saved model weights')
parser.add_argument('--model_type', type=str, default=None,
                    help='Model type to use. Options: CNNCatCross')
parser.add_argument('--dataset_type', type=str, default=None,
                    help='Dataset type to use. Options: real, simulated, combined')
parser.add_argument('--config_path', default=None,
                    help='Path to config file')
parser.add_argument('--fold', type=int, default=None,
                    help='Fold number to use (1-5)')
parser.add_argument('--alpha', type=float, default=0.1,
                    help='Significance level for conformal prediction (default: 0.1)')
args = parser.parse_args()

def load_config(config_path):
    """
    Load configuration from config yaml file. 
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_fold_datasets(config, fold_num):
    """
    Get calibration and test datasets for a specific fold.
    """
    data_config = config['data_configs']['loso_kfold']
    fold_key = f'fold{fold_num}'
    fold_config = data_config['folds'][fold_key]
    
    calibration_id = fold_config['val_ids'][0] 
    test_id = fold_config['test_ids'][0]
    
    # Real X-ray dataset 
    if args.dataset_type == "real":
        calibration_dataset = SpecimenSpecificDataset(
            data_root=data_config['data_root_real'],
            spec_id=calibration_id,
            image_side_size=data_config['image_side_size'],
            label_csv=data_config['label_csv_real'], 
            augmentation_factor=0, 
            is_real=True
        )
    
        test_dataset = SpecimenSpecificDataset(
            data_root=data_config['data_root_real'],
            spec_id=test_id,
            image_side_size=data_config['image_side_size'],
            label_csv=data_config['label_csv_real'], 
            augmentation_factor=0,
            is_real=True
        )
    
    # Simulated X-ray dataset
    if args.dataset_type == "simulated":
        calibration_dataset = SpecimenSpecificDataset(
            data_root=data_config['data_root_simulated'],
            spec_id=calibration_id,
            image_side_size=data_config['image_side_size'],
            label_csv=data_config['label_csv_simulated'], 
            augmentation_factor=0, 
            is_real=False
        )
    
        test_dataset = SpecimenSpecificDataset(
            data_root=data_config['data_root_simulated'],
            spec_id=test_id,
            image_side_size=data_config['image_side_size'],
            label_csv=data_config['label_csv_simulated'], 
            augmentation_factor=0,
            is_real=False
        )

    # Real and simulated X-ray datasets
    if args.dataset_type == "combined":
        real_calibration_dataset = SpecimenSpecificDataset(
            data_root=data_config['data_root_real'],
            spec_id=calibration_id, 
            image_side_size=data_config['image_side_size'],
            label_csv=data_config['label_csv_real'], 
            augmentation_factor=0,
            is_real=True 
        )

        simulated_calibration_dataset = SpecimenSpecificDataset(
            data_root=data_config['data_root_simulated'],
            spec_id=calibration_id,
            image_side_size=data_config['image_side_size'],
            label_csv=data_config['label_csv_simulated'],
            augmentation_factor=0, 
            is_real=False
            )

        real_test_dataset = SpecimenSpecificDataset(
            data_root=data_config['data_root_real'],
            spec_id=test_id,
            image_side_size=data_config['image_side_size'],
            label_csv=data_config['label_csv_real'],
            augmentation_factor=0,
            is_real=True 
        )

        simulated_test_dataset = SpecimenSpecificDataset(
            data_root=data_config['data_root_simulated'],
            spec_id=test_id,
            image_side_size=data_config['image_side_size'],
            label_csv=data_config['label_csv_simulated'],
            augmentation_factor=0,
            is_real=False 
        )

        calibration_dataset = CombinedDataset(real_calibration_dataset, simulated_calibration_dataset)
        test_dataset = CombinedDataset(real_test_dataset, simulated_test_dataset)
    
    
    return calibration_dataset, test_dataset

def get_nonconformity_scores(model, calibration_loader, device):
    """
    Compute nonconformity scores on the calibration set.
    """
    model.eval()
    scores = []
    true_labels = []
    
    with torch.no_grad():
        for xray_image, drr_image, target in calibration_loader:
            xray_image = xray_image.to(device)
            drr_image = drr_image.to(device)
            target = target.to(device).view(-1)
            
            with autocast():
                outputs = model(xray_image, drr_image)
            probs = torch.sigmoid(outputs).view(-1)
            
            batch_scores = torch.abs(probs - target).cpu().numpy()
            scores.extend(batch_scores.tolist())
            true_labels.extend(target.cpu().numpy().tolist())
    
    return np.array(scores), np.array(true_labels)

def get_prediction_sets(model, test_loader, epsilon, device):
    """
    Get prediction sets and model predictions for test dataset.
    """
    model.eval()
    prediction_sets = []
    true_labels = []
    model_predictions = []
    probabilities = []
    image_info = []
    
    with torch.no_grad():
        for batch_idx, (xray_image, drr_image, target) in enumerate(test_loader):
            xray_image = xray_image.to(device)
            drr_image = drr_image.to(device)
            
            for i in range(len(target)):
                idx = batch_idx * test_loader.batch_size + i
                if idx < len(test_loader.dataset):
                    if hasattr(test_loader.dataset, 'get_sample_info'):
                        info = test_loader.dataset.get_sample_info(idx)
                        image_info.append(info)
                    else:
                        sample = test_loader.dataset.samples[idx]
                        image_info.append({
                            'spec_id': test_loader.dataset.spec_id,
                            'proj_idx': sample['proj_idx'],
                            'sample_id': sample['sample_id']
                        })
            
            with autocast():
                outputs = model(xray_image, drr_image)
            
            probs = torch.sigmoid(outputs).view(-1).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            true_labels.extend(target.numpy().flatten().tolist())
            model_predictions.extend(preds.tolist())
            probabilities.extend(probs.tolist())
            
            for prob in probs:
                pred_set = []
                for y in [0, 1]:
                    if abs(prob - y) <= epsilon:
                        pred_set.append(y)
                prediction_sets.append(pred_set)
    
    return prediction_sets, np.array(true_labels), np.array(model_predictions), np.array(probabilities), image_info

def metrics(true_labels, predictions):
    """
    Compute metrics.
    """
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def main():
    logging.basicConfig(level=logging.INFO)
    
    config = load_config(args.config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    model = get_model_from_config(args.model_type, config).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    logging.info(f"Loaded model from {args.model_path}")
    
    calibration_dataset, test_dataset = get_fold_datasets(config, args.fold)
    
    fold_config = config['data_configs']['loso_kfold']['folds'][f'fold{args.fold}']
    calibration_id = fold_config['val_ids'][0]  
    test_id = fold_config['test_ids'][0]
    
    # Create dataloaders
    calibration_loader = DataLoader(calibration_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Get nonconformity scores on calibration set
    logging.info(f"Computing nonconformity scores on calibration set (Specimen {calibration_id})...")
    scores, calib_labels = get_nonconformity_scores(
        model, calibration_loader, device
    )
    
    # Compute threshold for desired coverage
    q = np.quantile(scores, 1 - args.alpha)
    logging.info(f"Nonconformity score threshold for {1-args.alpha:.1%} coverage: {q:.4f}")
    
    # Get prediction sets for test dataset
    logging.info(f"Generating prediction sets for test set (Specimen {test_id})...")
    prediction_sets, test_labels, predictions, probabilities, image_info = get_prediction_sets(
        model, test_loader, q, device
    )
    
    # Create DataFrame for CSV
    results_data = []
    for idx in range(len(test_labels)):
        results_data.append({
            'spec_id': image_info[idx]['spec_id'],
            'proj_idx': image_info[idx]['proj_idx'],
            'sample_id': image_info[idx]['sample_id'],
            'prediction': 'Accept' if predictions[idx] == 1 else 'Reject',
            'ground_truth': 'Accept' if test_labels[idx] == 1 else 'Reject', 
            'prob_accept': round(float(probabilities[idx]), 2),
            'prob_reject': round(1 - float(probabilities[idx]), 2),
            'certainty': 'Certain' if len(prediction_sets[idx]) == 1 else 'Uncertain'
        })
    
    results_df = pd.DataFrame(results_data)
    
    csv_filename = f'test_predictions_fold{args.fold}_{args.dataset_type}.csv'
    results_df.to_csv(csv_filename, index=False)
    logging.info(f'Results saved to {csv_filename}')
    
    # Compute metrics
    results = metrics(test_labels, predictions)
    
    # Compute coverage and efficiency 
    coverage = sum(
        test_labels[i] in pred_set 
        for i, pred_set in enumerate(prediction_sets)
    ) / len(test_labels)
    efficiency = np.mean([len(pred_set) for pred_set in prediction_sets])

    # Compute AUC
    auc_score = roc_auc_score(test_labels, probabilities)
    results['auc'] = auc_score
    
    # Log results
    logging.info(f'Conformal Prediction Results for Fold {args.fold}:')
    logging.info(f'Calibration Specimen: {calibration_id}')
    logging.info(f'Test Specimen: {test_id}')
    logging.info(f'Coverage: {coverage:.4f}')
    logging.info(f'Efficiency: {efficiency:.4f}')
    logging.info(f'Classification Metrics:')
    logging.info(f'True Positives: {results["true_positives"]}')
    logging.info(f'False Positives: {results["false_positives"]}')
    logging.info(f'True Negatives: {results["true_negatives"]}')
    logging.info(f'False Negatives: {results["false_negatives"]}')
    logging.info(f'Accuracy: {results["accuracy"]:.4f}')
    logging.info(f'Precision: {results["precision"]:.4f}')
    logging.info(f'Recall: {results["recall"]:.4f}')
    logging.info(f'F1 Score: {results["f1_score"]:.4f}')
    logging.info(f'AUC Score: {results["auc"]:.4f}')

if __name__ == '__main__':
    main()