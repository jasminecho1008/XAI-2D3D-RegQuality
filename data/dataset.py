import os
from glob import glob
import cv2
import torch
import numpy as np 
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset
import pandas as pd


def load_image(path):
    """Load an image as grayscale using OpenCV."""
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

class CombinedDataset(ConcatDataset):
    """
    Combine real and simulated projection datasets. 
    """
    def __init__(self, real_dataset, simulated_dataset):
        super().__init__([real_dataset, simulated_dataset])
        self.real_dataset = real_dataset
        self.simulated_dataset = simulated_dataset
        
        spec_ids = sorted(set(real_dataset.selected_spec_ids))  
        num_specs = len(spec_ids)
        
        total_accepts = (real_dataset.accept_count + simulated_dataset.accept_count)
        total_augmented = (total_accepts * real_dataset.augmentation_factor)
        total_rejects = (real_dataset.reject_count + simulated_dataset.reject_count)
        
        print(f"Dataset statistics across {num_specs} specimens {spec_ids}:")
        print(f"- Original accepts: {total_accepts}")
        print(f"- Augmented accepts: {total_augmented}")
        print(f"- Total accepts: {total_accepts + total_augmented}")
        print(f"- Rejects: {total_rejects}")
        print(f"- Total samples: {len(self)}")
        
    def get_sample_info(self, idx):
        if idx < 0:
            idx += len(self)
        if idx < self.real_size:
            return self.real_dataset.get_sample_info(idx)
        else:
            return self.simulated_dataset.get_sample_info(idx - self.real_size)

    @property
    def real_size(self):
        return len(self.real_dataset)
    
    @property
    def simulated_size(self):
        return len(self.simulated_dataset)
    
    def __str__(self):
        return f"CombinedDataset(real_samples={self.real_size}, simulated_samples={self.simulated_size})"

class ImageAugmentation:
    """Non-geometric image augmentation."""
    @staticmethod
    def random_brightness_contrast(image, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)):
        """Apply random brightness and contrast augmentation."""
        img_float = image.astype(float)
        
        # Random brightness
        brightness = np.random.uniform(*brightness_range)
        img_float *= brightness
        
        # Random contrast
        contrast = np.random.uniform(*contrast_range)
        mean = np.mean(img_float)
        img_float = (img_float - mean) * contrast + mean
        
        return np.clip(img_float, 0, 255).astype(np.uint8)
    
    @staticmethod
    def random_noise(image, noise_range=(0, 10)):
        """Apply random noise augmentation."""
        noise = np.random.normal(0, np.random.uniform(*noise_range), image.shape)
        noisy_img = image + noise
        return np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    @staticmethod
    def random_blur(image, kernel_range=(1, 3)):
        """Apply random blur augmentation."""
        kernel_size = np.random.randint(*kernel_range) * 2 + 1  
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


class SpecimenSpecificDataset(Dataset):
    """
    Dataset for specimen-specific experiments.

    Directory structure:
      data_root/
      ├── Image_Poses_mTRE_binary.csv   # CSV with columns: spec_id, proj_idx, sample_id, binary, ...
      └── specimen_folder/              # e.g., "17-1882"
             ├──  XXX/                  # Projection folder named as a 3-digit number derived from proj_idx, e.g., "001", "002", etc.
             │      ├── xray.png        # Target image.
             │      └── DRR/            # Folder containing DRR images for that projection.
             │             └── drr_remap_YYY.png   # DRR image where YYY is a 3-digit number derived from sample_id, e.g., "000", "001", etc.
             └── ...

    Each CSV row defines one sample by specifying:
      - spec_id: should match the provided specimen folder name.
      - proj_idx: used to determine the projection folder name (formatted as f"{proj_idx:03d}").
      - sample_id: used to determine the DRR file name (formatted as f"{sample_id:03d}").
      - binary: the binary registration quality label.

    Args:
      data_root (str): Path to the root directory containing the CSV file and the specimen folder.
      spec_id (str): The specific specimen ID to use (e.g., "17-1882").
      image_side_size (int): Size to which all images will be resized.
      label_csv (str): Name of the CSV file (located in data_root).
      transform (callable, optional): A torchvision transform to apply to the images.
      reject_threshold (float): Maximum allowable rejection rate for a projection.
      augmentation_factor (int): Number of augmented copies to create for each accepted sample. 
      is_real (bool): Flag to indicate whether dataset contains real or simulated projections. 
    """
    def __init__(self, data_root, spec_id, image_side_size=64, label_csv='Image_Poses_mTRE_binary.csv', 
                 transform=None, reject_threshold=0.9, augmentation_factor=1, is_real=False):
        self.data_root = data_root
        self.spec_id = spec_id
        self.augmenter = ImageAugmentation()
        self.augmentation_factor = augmentation_factor
        self.selected_spec_ids = [spec_id]  
        self.is_real = is_real 

        # The specimen folder is assumed to be data_root/spec_id
        self.specimen_folder = os.path.join(data_root, spec_id)
        if not os.path.isdir(self.specimen_folder):
            raise FileNotFoundError(f"Specimen folder {spec_id} not found in {data_root}")

        # Load the CSV file from data_root
        label_csv_path = os.path.join(data_root, label_csv)
        if not os.path.exists(label_csv_path):
            raise FileNotFoundError(f"Label CSV not found at {label_csv_path}")
        
        df = pd.read_csv(label_csv_path)
        df_specimen = df[df['spec_id'] == spec_id].copy()
        
        # Compute rejection rate per projection
        df_specimen['is_reject'] = df_specimen['binary'] == 0
        proj_stats = df_specimen.groupby('proj_idx')['is_reject'].mean()
        
        # Filter out projections that exceed reject_threshold
        valid_projections = proj_stats[proj_stats <= reject_threshold].index
        df_filtered = df_specimen[df_specimen['proj_idx'].isin(valid_projections)]
        
        # Build a list of samples with augmentation for accept cases
        self.samples = []
        self.accept_count = 0
        self.reject_count = 0
        
        for _, row in df_filtered.iterrows():
            is_accept = row['binary'] == 1
            base_sample = {
                'proj_idx': int(row['proj_idx']),
                'sample_id': int(row['sample_id']),
                'binary': int(row['binary']),
                'is_accept': is_accept,
                'is_augmented': False
            }
            
            if is_accept:
                self.accept_count += 1
                self.samples.append(base_sample)
                for aug_idx in range(augmentation_factor):
                    aug_sample = base_sample.copy()
                    aug_sample['is_augmented'] = True
                    aug_sample['aug_idx'] = aug_idx
                    self.samples.append(aug_sample)
            else:
                self.reject_count += 1
                self.samples.append(base_sample)

        self.image_side_size = image_side_size
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_side_size, image_side_size)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def normalize_xray(self, image):
        """Normalize real X-ray image to [0,1] range."""
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        image = (image - image.min()) / (image.max() - image.min())
        return image 

    def augment_images(self, target_img, drr_img):
        """Apply random augmentations to create a new pair."""
        target_img = target_img.copy()
        drr_img = drr_img.copy()

        if self.is_real:
            target_img = (target_img * 255).astype(np.uint8)
        
        # Apply brightness and contrast augmentation
        if np.random.random() < 0.8:
            target_img = self.augmenter.random_brightness_contrast(target_img)
            drr_img = self.augmenter.random_brightness_contrast(drr_img)
        
        # Apply noise augmentation
        if np.random.random() < 0.6:
            target_img = self.augmenter.random_noise(target_img)
            drr_img = self.augmenter.random_noise(drr_img)
        
        # Apply blur augmentation
        if np.random.random() < 0.4:
            target_img = self.augmenter.random_blur(target_img)
            drr_img = self.augmenter.random_blur(drr_img)

        if self.is_real:
            target_img = target_img.astype(np.float32) / 255.0
            
        return target_img, drr_img
    
    def get_sample_info(self, idx):
        sample = self.samples[idx]
        return {
            'spec_id': self.spec_id,
            'proj_idx': int(sample['proj_idx']),
            'sample_id': int(sample['sample_id']),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        proj_idx = sample['proj_idx']
        sample_id = sample['sample_id']
        is_accept = sample['is_accept']
        is_augmented = sample['is_augmented']

        # Construct folder name from proj_idx. Assume projection folder name is a zero-padded 3-digit string.
        proj_folder = f"{proj_idx:03d}"
        # Construct DRR file name from sample_id. Assume DRR file name is "drr_remap_{sample_id:03d}.png"
        drr_file = f"drr_remap_{sample_id:03d}.png"

        # Load target image
        target_path = os.path.join(self.specimen_folder, proj_folder, 'xray.png')
        target_img = load_image(target_path)
        if target_img is None:
            return self.__getitem__((idx + 1) % len(self.samples))
        
        if self.is_real:
            target_img = self.normalize_xray(target_img)

        # Target image is at: specimen_folder/proj_folder/xray.png
        drr_path = os.path.join(self.specimen_folder, proj_folder, 'DRR', drr_file)
        drr_img = load_image(drr_path)
        if drr_img is None:
            # Skip this sample if DRR image is missing
            # print(f"Warning: DRR image not found at {drr_path}. Skipping sample.")
            return self.__getitem__((idx + 1) % len(self.samples))

        # Apply augmentation for accept cases
        if is_accept and is_augmented:
            target_img, drr_img = self.augment_images(target_img, drr_img)

        if self.is_real:
            target_img = torch.from_numpy(target_img).unsqueeze(0)  
            target_img = transforms.Resize((self.image_side_size, self.image_side_size))(target_img)
            drr_img = self.transform(drr_img)
        else:
            target_img = self.transform(target_img)
            drr_img = self.transform(drr_img)

        label = torch.tensor(float(is_accept))

        return target_img, drr_img, label

    
class LOSODataset(Dataset):
    """
    Dataset for leave-one-specimen-out experiments. 

    Directory structure:
      data_root/
      ├── Image_Poses_mTRE_binary.csv   # CSV with columns: spec_id, proj_idx, sample_id, binary, ...
      └── specimen_folders/              # e.g., "17-1882", "18-0725", etc. 
             ├──  XXX/                  # Projection folder named as a 3-digit number derived from proj_idx, e.g., "001", "002", etc.
             │      ├── xray.png        # Target image.
             │      └── DRR/            # Folder containing DRR images for that projection.
             │             └── drr_remap_YYY.png   # DRR image where YYY is a 3-digit number derived from sample_id, e.g., "000", "001", etc.
             └── ...

    Each CSV row defines one sample by specifying:
      - spec_id: should match the provided specimen folder name.
      - proj_idx: used to determine the projection folder name (formatted as f"{proj_idx:03d}").
      - sample_id: used to determine the DRR file name (formatted as f"{sample_id:03d}").
      - binary: the binary registration quality label.


    Args: 
        data_root (str): Path to the root directory containing the CSV file and specimen folders.
        selected_spec_ids (list): List to the specimen IDs to include in this dataset split. 
        image_side_size (int): Size to which all images will be resized. 
        label_csv (str): Name of the CSV file (located in data_root).
        transform (callable, optional): A torchvision transform to apply to the images.
        reject_threshold (float): Maximum allowable rejection rate for a projection.
        augmentation_factor (int): Number of augmented copies to create for each accepted sample. 
        is_real (bool): Flag to indicate whether dataset contains real or simulated projections. 
    """
    def __init__(self, data_root, selected_spec_ids, image_side_size=64, label_csv="Image_Poses_mTRE_binary.csv", 
                transform=None, reject_threshold=0.9, augmentation_factor=None, is_real=False):
        self.data_root = data_root
        self.selected_spec_ids = selected_spec_ids
        self.augmenter = ImageAugmentation()
        self.augmentation_factor = augmentation_factor  
        self.is_real = is_real 

        # Load the CSV from data_root
        label_csv_path = os.path.join(data_root, label_csv)
        if not os.path.exists(label_csv_path):
            raise FileNotFoundError(f"Label CSV not found at {label_csv_path}")
        
        df = pd.read_csv(label_csv_path)
        df_filtered = df[df['spec_id'].isin(selected_spec_ids)].copy()
        
        # Compute rejection rate per specimen/projection 
        df_filtered['is_reject'] = df_filtered['binary'] == 0  
        proj_stats = df_filtered.groupby(['spec_id', 'proj_idx'])['is_reject'].mean()
        
        # Filter out projections that exceed reject_threshold
        valid_projection = proj_stats[proj_stats <= reject_threshold].reset_index()
        
        # Build a list of samples with augmentation for accept cases
        self.samples = []
        self.accept_count = 0  
        self.reject_count = 0  
        
        for spec_id in selected_spec_ids:
            specimen_folder = os.path.join(data_root, spec_id)
            if not os.path.isdir(specimen_folder):
                raise FileNotFoundError(f"Specimen folder {spec_id} not found in {data_root}")
            
            valid_projs = valid_projection[valid_projection['spec_id'] == spec_id]['proj_idx']
            
            df_specimen = df_filtered[(df_filtered['spec_id'] == spec_id) & (df_filtered['proj_idx'].isin(valid_projs))]
            
            for _, row in df_specimen.iterrows():
                is_accept = row['binary'] == 1  
                base_sample = {
                    'spec_id': spec_id,
                    'proj_idx': int(row['proj_idx']),
                    'sample_id': int(row['sample_id']),
                    'binary': int(row['binary']),
                    'is_accept': is_accept,
                    'is_augmented': False
                }
                
                if is_accept:
                    self.accept_count += 1
                    self.samples.append(base_sample)
                    for aug_idx in range(augmentation_factor):
                        aug_sample = base_sample.copy()
                        aug_sample['is_augmented'] = True
                        aug_sample['aug_idx'] = aug_idx
                        self.samples.append(aug_sample)
                else:
                    self.reject_count += 1
                    self.samples.append(base_sample)

        self.image_side_size = image_side_size
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_side_size, image_side_size)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def normalize_xrays(self, image):
        """Normalize real X-ray image to [0,1] range."""
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        image = (image - image.min()) / (image.max() - image.min())
        return image 

    def augment_images(self, target_img, drr_img):
        """Apply random augmentations to create a new pair."""
        target_img = target_img.copy()
        drr_img = drr_img.copy()

        if self.is_real: 
            target_img = (target_img * 255).astype(np.uint8)
        
        # Apply brightness and contrast augmentation
        if np.random.random() < 0.8:  
            target_img = self.augmenter.random_brightness_contrast(target_img)
            drr_img = self.augmenter.random_brightness_contrast(drr_img)
        
        # Apply noise augmentation
        if np.random.random() < 0.6:  
            target_img = self.augmenter.random_noise(target_img)
            drr_img = self.augmenter.random_noise(drr_img)
        
        # Apply blur augmentation
        if np.random.random() < 0.4:  
            target_img = self.augmenter.random_blur(target_img)
            drr_img = self.augmenter.random_blur(drr_img)

        if self.is_real: 
            target_img = target_img.astype(np.float32) / 255.0 
            
        return target_img, drr_img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        spec_id = sample['spec_id']
        proj_idx = sample['proj_idx']
        sample_id = sample['sample_id']
        is_accept = sample['is_accept']
        is_augmented = sample['is_augmented']

        specimen_folder = os.path.join(self.data_root, spec_id)
        # Construct folder name from proj_idx. Assume projection folder name is a zero-padded 3-digit string.
        proj_folder = f"{proj_idx:03d}"
        # Construct DRR file name from sample_id. Assume DRR file name is "drr_remap_{sample_id:03d}.png" 
        drr_file = f"drr_remap_{sample_id:03d}.png"

        # Target image is at: specimen_folder/proj_folder/xray.png
        target_path = os.path.join(specimen_folder, proj_folder, 'xray.png')
        target_img = load_image(target_path)
        if target_img is None:
            # Skip this sample if target image is missing
            # print(f"Warning: Target image not found at {target_path}. Skipping sample.")
            return self.__getitem__((idx + 1) % len(self.samples))
        
        if self.is_real: 
            target_img = self.normalize_xrays(target_img)

        # DRR image is at: specimen_folder/proj_folder/DRR/drr_file
        drr_path = os.path.join(specimen_folder, proj_folder, 'DRR', drr_file)
        drr_img = load_image(drr_path)
        if drr_img is None:
            return self.__getitem__((idx + 1) % len(self.samples))

        # Apply augmentation for accept cases
        if is_accept and is_augmented:
            target_img, drr_img = self.augment_images(target_img, drr_img)

        if self.is_real:
            target_img = torch.from_numpy(target_img).unsqueeze(0)  
            target_img = transforms.Resize((self.image_side_size, self.image_side_size))(target_img)
            drr_img = self.transform(drr_img)
        else:
            target_img = self.transform(target_img)
            drr_img = self.transform(drr_img)

        label = torch.tensor(float(is_accept))

        return target_img, drr_img, label