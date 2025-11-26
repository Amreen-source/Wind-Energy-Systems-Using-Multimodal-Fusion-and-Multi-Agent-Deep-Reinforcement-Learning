"""
Text Processor for Maintenance Logs
Processes maintenance logs using BERT/transformers
"""

import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from typing import List, Dict
import pickle
from pathlib import Path
import random

from config import config


class MaintenanceTextProcessor:
    """Process maintenance logs and generate text embeddings"""
    
    def __init__(self):
        """Initialize text processor"""
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        self.processed_data_path = config.PROCESSED_DATA_DIR / "maintenance_logs.pkl"
        
        # Load BERT model
        print(f"Loading text model: {config.TEXT_MODEL_NAME}")
        self.tokenizer = BertTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        self.model = BertModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load or generate maintenance logs
        if self.processed_data_path.exists():
            print(f"Loading processed maintenance logs from {self.processed_data_path}")
            self.load_processed_data()
        else:
            print("Generating synthetic maintenance logs...")
            self.generate_synthetic_logs()
    
    def generate_synthetic_logs(self):
        """Generate realistic synthetic maintenance logs"""
        
        # Template-based generation for realism
        components = ['blade', 'gearbox', 'generator', 'tower', 'nacelle']
        
        normal_templates = [
            "Routine inspection of {component}. All systems operational.",
            "{component} operating within normal parameters. No issues detected.",
            "Scheduled maintenance on {component} completed successfully.",
            "Visual inspection of {component} shows no signs of wear.",
            "{component} temperature and vibration levels normal.",
        ]
        
        warning_templates = [
            "Minor vibration detected in {component}. Monitoring required.",
            "{component} temperature slightly elevated. Schedule inspection.",
            "Unusual noise from {component} during operation. Investigate soon.",
            "Small crack observed on {component} surface. Non-critical.",
            "{component} showing early signs of wear. Plan preventive maintenance.",
        ]
        
        critical_templates = [
            "URGENT: Excessive vibration in {component}. Immediate inspection required.",
            "CRITICAL: {component} temperature above safe threshold. Shutdown recommended.",
            "FAILURE DETECTED: {component} malfunction. Corrective maintenance needed.",
            "ALERT: Significant damage to {component}. Replace parts immediately.",
            "CRITICAL: {component} performance degradation. Risk of failure high.",
        ]
        
        # Generate logs for each turbine
        self.maintenance_logs = {}
        
        for turbine_id in range(config.NUM_TURBINES):
            turbine_logs = []
            
            # Generate logs for simulation period (one log every few days)
            num_logs = config.SIMULATION_DAYS // 7  # Weekly logs
            
            for _ in range(num_logs):
                # Choose component
                component = random.choice(components)
                
                # Choose severity (most logs are normal)
                severity_choice = random.choices(
                    ['normal', 'warning', 'critical'],
                    weights=[0.7, 0.25, 0.05]
                )[0]
                
                if severity_choice == 'normal':
                    template = random.choice(normal_templates)
                    severity = 0.0
                elif severity_choice == 'warning':
                    template = random.choice(warning_templates)
                    severity = 0.5
                else:
                    template = random.choice(critical_templates)
                    severity = 1.0
                
                text = template.format(component=component)
                
                turbine_logs.append({
                    'text': text,
                    'component': component,
                    'severity': severity,
                    'embedding': None  # Will be computed
                })
            
            self.maintenance_logs[turbine_id] = turbine_logs
        
        # Compute embeddings for all logs
        self._compute_embeddings()
        self.save_processed_data()
        
        total_logs = sum(len(logs) for logs in self.maintenance_logs.values())
        print(f"✓ Generated {total_logs} synthetic maintenance logs")
    
    def _compute_embeddings(self):
        """Compute BERT embeddings for all logs"""
        
        print("Computing text embeddings...")
        
        with torch.no_grad():
            for turbine_id, logs in self.maintenance_logs.items():
                for log in logs:
                    if log['embedding'] is None:
                        embedding = self.encode_text(log['text'])
                        log['embedding'] = embedding
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using BERT"""
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=config.MAX_TEXT_LENGTH,
            padding='max_length',
            truncation=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        return embedding
    
    def get_recent_logs(self, turbine_id: int, num_logs: int = 5) -> List[Dict]:
        """Get recent maintenance logs for a turbine"""
        
        logs = self.maintenance_logs.get(turbine_id, [])
        return logs[-num_logs:] if logs else []
    
    def get_text_features(self, turbine_id: int) -> np.ndarray:
        """
        Get aggregated text features for RL state
        
        Args:
            turbine_id: Turbine index
        
        Returns:
            Aggregated text embedding
        """
        recent_logs = self.get_recent_logs(turbine_id, num_logs=3)
        
        if not recent_logs:
            # Return zero embedding if no logs
            return np.zeros(config.TEXT_EMBED_DIM, dtype=np.float32)
        
        # Average recent embeddings
        embeddings = [log['embedding'] for log in recent_logs]
        avg_embedding = np.mean(embeddings, axis=0)
        
        # Add severity information
        avg_severity = np.mean([log['severity'] for log in recent_logs])
        
        # Concatenate
        features = np.concatenate([avg_embedding, [avg_severity]])
        
        return features.astype(np.float32)
    
    def save_processed_data(self):
        """Save processed logs to disk"""
        with open(self.processed_data_path, 'wb') as f:
            pickle.dump(self.maintenance_logs, f)
        print(f"✓ Saved maintenance logs to {self.processed_data_path}")
    
    def load_processed_data(self):
        """Load processed logs from disk"""
        with open(self.processed_data_path, 'rb') as f:
            self.maintenance_logs = pickle.load(f)
        
        total_logs = sum(len(logs) for logs in self.maintenance_logs.values())
        print(f"✓ Loaded {total_logs} maintenance logs")
    
    def get_statistics(self) -> Dict:
        """Get statistics about maintenance logs"""
        total_logs = sum(len(logs) for logs in self.maintenance_logs.values())
        
        all_severities = []
        for logs in self.maintenance_logs.values():
            all_severities.extend([log['severity'] for log in logs])
        
        stats = {
            'total_logs': total_logs,
            'logs_per_turbine': {k: len(v) for k, v in self.maintenance_logs.items()},
            'embedding_dim': config.TEXT_EMBED_DIM,
            'avg_severity': np.mean(all_severities) if all_severities else 0,
            'critical_logs': sum(1 for s in all_severities if s > 0.8)
        }
        
        return stats


# Test the text processor
if __name__ == "__main__":
    print("Testing Maintenance Text Processor...")
    print("=" * 60)
    
    processor = MaintenanceTextProcessor()
    stats = processor.get_statistics()
    
    print("\nText Statistics:")
    for key, value in stats.items():
        if key != 'logs_per_turbine':
            print(f"  {key}: {value}")
    
    print("\nSample logs for turbine 0:")
    logs = processor.get_recent_logs(0, num_logs=3)
    for i, log in enumerate(logs, 1):
        print(f"  Log {i}: {log['text'][:60]}... (severity: {log['severity']})")
    
    print("\nText features for RL:")
    features = processor.get_text_features(0)
    print(f"  Shape: {features.shape}")
    print(f"  Mean: {features.mean():.4f}")
    
    print("\n✓ Text Processor test passed!")
