"""
Image Processor for Wind Turbine Blade Inspection
Processes drone/camera imagery using Vision Transformers
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pickle
from tqdm import tqdm
from transformers import ViTModel, ViTImageProcessor

from config import config


class BladeImageProcessor:
    """Process wind turbine blade inspection images"""
    
    def __init__(self, images_path: Path = None, labels_path: Path = None):
        """
        Initialize image processor
        
        Args:
            images_path: Path to blade inspection images
            labels_path: Path to defect labels
        """
        self.images_path = images_path or config.BLADE_IMAGES_PATH
        self.labels_path = labels_path or config.BLADE_LABELS_PATH
        self.processed_data_path = config.PROCESSED_DATA_DIR / "blade_images_processed.pkl"
        
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        
        # Load Vision Transformer
        print(f"Loading Vision Transformer: {config.VISION_MODEL_NAME}")
        self.vit_processor = ViTImageProcessor.from_pretrained(config.VISION_MODEL_NAME)
        self.vit_model = ViTModel.from_pretrained(config.VISION_MODEL_NAME)
        self.vit_model = self.vit_model.to(self.device)
        self.vit_model.eval()
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Augmentation for training
        self.augment_transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]) if config.IMAGE_AUGMENTATION else self.transform
        
        # Load and process images
        if self.processed_data_path.exists():
            print(f"Loading processed images from {self.processed_data_path}")
            self.load_processed_data()
        else:
            print(f"Processing blade images from {self.images_path}")
            self.process_images()
    
    def process_images(self):
        """Process all blade inspection images"""
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(self.images_path.glob(f"*{ext}")))
            image_files.extend(list(self.images_path.glob(f"*{ext.upper()}")))
        
        if not image_files:
            print("⚠️  No blade images found. Generating synthetic image features...")
            self.generate_synthetic_image_features()
            return
        
        print(f"Found {len(image_files)} blade inspection images")
        
        # Load labels if available
        label_data = self._load_labels()
        
        # Process images in batches
        batch_size = 32
        all_features = []
        all_labels = []
        all_image_ids = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(image_files), batch_size), desc="Processing images"):
                batch_files = image_files[i:i + batch_size]
                batch_images = []
                batch_ids = []
                
                for img_path in batch_files:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img_tensor = self.transform(img)
                        batch_images.append(img_tensor)
                        batch_ids.append(img_path.stem)
                    except Exception as e:
                        print(f"⚠️  Error loading {img_path}: {e}")
                        continue
                
                if not batch_images:
                    continue
                
                # Extract features using ViT
                batch_tensor = torch.stack(batch_images).to(self.device)
                outputs = self.vit_model(pixel_values=batch_tensor)
                features = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token
                
                all_features.append(features)
                all_image_ids.extend(batch_ids)
                
                # Get labels for this batch
                for img_id in batch_ids:
                    label = label_data.get(img_id, {'defect': 0, 'severity': 0.0})
                    all_labels.append(label)
        
        if not all_features:
            print("⚠️  No images could be processed. Generating synthetic features...")
            self.generate_synthetic_image_features()
            return
        
        # Combine all features
        self.image_features = np.vstack(all_features)
        self.image_labels = all_labels
        self.image_ids = all_image_ids
        
        # Organize by turbine (assume filename format: turbine_XX_...)
        self.turbine_images = self._organize_by_turbine()
        
        self.save_processed_data()
        print(f"✓ Processed {len(self.image_features)} images")
    
    def _load_labels(self) -> Dict:
        """Load defect labels if available"""
        label_data = {}
        
        if not self.labels_path.exists():
            return label_data
        
        # Try different label formats
        label_files = list(self.labels_path.glob("*.txt")) + \
                     list(self.labels_path.glob("*.csv")) + \
                     list(self.labels_path.glob("*.json"))
        
        for label_file in label_files:
            try:
                if label_file.suffix == '.txt':
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                img_id = parts[0]
                                defect = int(parts[1]) if len(parts) > 1 else 0
                                severity = float(parts[2]) if len(parts) > 2 else 0.0
                                label_data[img_id] = {'defect': defect, 'severity': severity}
                
                elif label_file.suffix == '.csv':
                    import pandas as pd
                    df = pd.read_csv(label_file)
                    for _, row in df.iterrows():
                        img_id = str(row.get('image_id', row.iloc[0]))
                        defect = int(row.get('defect', row.get('label', 0)))
                        severity = float(row.get('severity', 0.0))
                        label_data[img_id] = {'defect': defect, 'severity': severity}
                
                elif label_file.suffix == '.json':
                    import json
                    with open(label_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            label_data.update(data)
            
            except Exception as e:
                print(f"⚠️  Error loading labels from {label_file}: {e}")
                continue
        
        return label_data
    
    def _organize_by_turbine(self) -> Dict[int, List[int]]:
        """Organize images by turbine ID"""
        turbine_images = {i: [] for i in range(config.NUM_TURBINES)}
        
        for idx, img_id in enumerate(self.image_ids):
            # Try to extract turbine ID from filename
            # Common patterns: "turbine_05_...", "T05_...", "05_..."
            import re
            match = re.search(r'[Tt]?_?(\d{1,2})', img_id)
            if match:
                turbine_id = int(match.group(1)) % config.NUM_TURBINES
            else:
                # Distribute evenly if no ID found
                turbine_id = idx % config.NUM_TURBINES
            
            turbine_images[turbine_id].append(idx)
        
        return turbine_images
    
    def generate_synthetic_image_features(self):
        """Generate synthetic image features when real images unavailable"""
        
        print(f"Generating synthetic image features for {config.NUM_TURBINES} turbines...")
        
        # Generate realistic ViT features (768-dim)
        num_images_per_turbine = config.MAX_IMAGES_PER_TURBINE
        total_images = config.NUM_TURBINES * num_images_per_turbine
        
        # Generate features with some structure (not pure random)
        base_features = np.random.randn(config.NUM_TURBINES, config.VISION_EMBED_DIM) * 0.5
        
        all_features = []
        all_labels = []
        
        for turbine_id in range(config.NUM_TURBINES):
            for img_idx in range(num_images_per_turbine):
                # Add turbine-specific pattern + random noise
                feature = base_features[turbine_id] + np.random.randn(config.VISION_EMBED_DIM) * 0.3
                
                # Normalize to unit sphere (common for embeddings)
                feature = feature / (np.linalg.norm(feature) + 1e-8)
                
                all_features.append(feature)
                
                # Generate synthetic labels
                defect = np.random.choice([0, 1], p=[0.8, 0.2])  # 20% defect rate
                severity = np.random.beta(2, 5) if defect else 0.0  # Skewed towards low severity
                all_labels.append({'defect': defect, 'severity': severity})
        
        self.image_features = np.array(all_features)
        self.image_labels = all_labels
        self.image_ids = [f"synthetic_{i}" for i in range(total_images)]
        
        # Organize by turbine
        self.turbine_images = {
            i: list(range(i * num_images_per_turbine, (i + 1) * num_images_per_turbine))
            for i in range(config.NUM_TURBINES)
        }
        
        self.save_processed_data()
        print(f"✓ Generated {len(self.image_features)} synthetic image features")
    
    def get_turbine_images(self, turbine_id: int, 
                          max_images: int = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Get image features and labels for a specific turbine
        
        Args:
            turbine_id: Turbine index
            max_images: Maximum number of images to return
        
        Returns:
            features: (N, embed_dim) array
            labels: List of label dicts
        """
        if max_images is None:
            max_images = config.MAX_IMAGES_PER_TURBINE
        
        image_indices = self.turbine_images.get(turbine_id, [])
        
        if not image_indices:
            # Return zero features if no images
            return np.zeros((1, config.VISION_EMBED_DIM)), [{'defect': 0, 'severity': 0.0}]
        
        # Select most recent images (or random sample)
        selected_indices = image_indices[-max_images:] if len(image_indices) >= max_images else image_indices
        
        features = self.image_features[selected_indices]
        labels = [self.image_labels[i] for i in selected_indices]
        
        return features, labels
    
    def get_aggregated_features(self, turbine_id: int) -> np.ndarray:
        """
        Get aggregated image features for a turbine (for RL state)
        
        Args:
            turbine_id: Turbine index
        
        Returns:
            Aggregated feature vector (embed_dim,)
        """
        features, labels = self.get_turbine_images(turbine_id)
        
        # Aggregate by averaging (could also use attention pooling)
        aggregated = features.mean(axis=0)
        
        # Add defect information
        defect_rate = np.mean([l['defect'] for l in labels])
        avg_severity = np.mean([l['severity'] for l in labels])
        
        # Append as additional features
        aggregated = np.concatenate([aggregated, [defect_rate, avg_severity]])
        
        return aggregated.astype(np.float32)
    
    def save_processed_data(self):
        """Save processed data to disk"""
        data = {
            'features': self.image_features,
            'labels': self.image_labels,
            'image_ids': self.image_ids,
            'turbine_images': self.turbine_images
        }
        with open(self.processed_data_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Saved processed images to {self.processed_data_path}")
    
    def load_processed_data(self):
        """Load processed data from disk"""
        with open(self.processed_data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.image_features = data['features']
        self.image_labels = data['labels']
        self.image_ids = data['image_ids']
        self.turbine_images = data['turbine_images']
        
        print(f"✓ Loaded {len(self.image_features)} processed images")
    
    def get_statistics(self) -> Dict:
        """Get statistics about processed images"""
        total_defects = sum(1 for label in self.image_labels if label['defect'] == 1)
        avg_severity = np.mean([label['severity'] for label in self.image_labels])
        
        stats = {
            'total_images': len(self.image_features),
            'feature_dim': self.image_features.shape[1],
            'num_turbines': len(self.turbine_images),
            'images_per_turbine': {k: len(v) for k, v in self.turbine_images.items()},
            'total_defects': total_defects,
            'defect_rate': total_defects / len(self.image_labels),
            'avg_severity': avg_severity
        }
        
        return stats


# Test the processor
if __name__ == "__main__":
    print("Testing Blade Image Processor...")
    print("=" * 60)
    
    processor = BladeImageProcessor()
    stats = processor.get_statistics()
    
    print("\nImage Statistics:")
    for key, value in stats.items():
        if key != 'images_per_turbine':
            print(f"  {key}: {value}")
    
    print("\nSample features for turbine 0:")
    features, labels = processor.get_turbine_images(0, max_images=3)
    print(f"  Features shape: {features.shape}")
    print(f"  Labels: {labels}")
    
    print("\nAggregated features for RL:")
    agg_features = processor.get_aggregated_features(0)
    print(f"  Shape: {agg_features.shape}")
    print(f"  Mean: {agg_features.mean():.4f}")
    print(f"  Std: {agg_features.std():.4f}")
    
    print("\n✓ Image Processor test passed!")
