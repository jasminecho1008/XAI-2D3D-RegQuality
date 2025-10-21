import logging
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import numpy as np
import yaml
import os
import cv2

from models import get_model_from_config
from data.dataset import SpecimenSpecificDataset, CombinedDataset

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=None,
                    help='Path to saved model weights')
parser.add_argument('--model_type', type=str, default=None,
                    help='Model type to use (overrides config). Options: CNNCatCross')
parser.add_argument('--dataset_type', type=str, default=None,
                    help='Dataset type to use. Options: real, simulated, combined')
parser.add_argument('--config_path', default='configs/config.yaml',
                    help='Path to config file')
parser.add_argument('--fold', type=int, default=None,
                    help='Fold number to use (1-5)')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Output directory to save Grad-CAM heatmaps')
args = parser.parse_args()

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None
        
        target_layer.register_forward_hook(self.save_features)
        target_layer.register_full_backward_hook(self.save_gradients)
    
    def save_features(self, module, inputs, output):
        self.features = output.detach()
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, xray_image, drr_image):
        self.model.eval()
        
        with torch.set_grad_enabled(True): 
            with autocast():
                output = self.model(xray_image, drr_image)
            
            self.model.zero_grad()
            output_scalar = output.sum()  
            output_scalar.backward()
            
            gradients = self.gradients.mean(dim=[2, 3], keepdim=True)
            cam = torch.sum(gradients * self.features, dim=1, keepdim=True)
            cam = F.relu(cam)  
            
            cam = F.interpolate(cam, size=xray_image.shape[2:], mode='bilinear', align_corners=False)
            cam_min = cam.amin(dim=(2,3), keepdim=True)
            cam_max = cam.amax(dim=(2,3), keepdim=True)
            cam = (cam - cam_min) / (cam_max - cam_min).clamp_min(torch.finfo(cam.dtype).eps) 
            
            return cam.detach()

def load_config(config_path):
    """
    Load configuration from config yaml file. 
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_fold_datasets(config, fold_num):
    """
    Get test dataset for a specific fold.
    """
    data_config = config['data_configs']['loso_kfold']
    fold_key = f'fold{fold_num}'
    fold_config = data_config['folds'][fold_key]
    
    # Get specimen IDs for test set 
    test_id = fold_config['test_ids'][0]
    
    # Real X-ray dataset 
    if args.dataset_type == "real":
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

        test_dataset = CombinedDataset(real_test_dataset, simulated_test_dataset)
    
    return test_dataset

def find_last_conv_layer(model):
    last_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise ValueError("Could not find a convolutional layer")
    return last_conv

def save_heatmap(cam_norm, info, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    heatmap = (np.clip(cam_norm, 0, 1) * 255).astype(np.uint8)  
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    filename = f"{info['spec_id']}_{int(info['proj_idx']):03d}_{int(info['sample_id']):03d}.png"
    cv2.imwrite(os.path.join(output_dir, filename), colored)

def main():
    logging.basicConfig(level=logging.INFO)

    config = load_config(args.config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load model
    model = get_model_from_config(args.model_type, config).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    logging.info(f"Loaded model from {args.model_path}")
    
    # Get test dataset for the specified fold
    test_dataset = get_fold_datasets(config, args.fold)

    # Create dataloader
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    output_dir = args.output_dir 
    os.makedirs(output_dir, exist_ok=True)

    target_layer = find_last_conv_layer(model)
    cam = GradCAM(model, target_layer)

    for batch_idx, (xray, drr, _) in enumerate(test_loader):
        xray = xray.to(device)
        drr = drr.to(device)
        cams = cam.generate_cam(xray, drr)

        base = batch_idx * test_loader.batch_size
        for i in range(cams.size(0)):
            info = test_loader.dataset.get_sample_info(base + i)
            save_heatmap(cams[i,0].cpu().numpy(), info, output_dir)

if __name__ == '__main__':
    main()