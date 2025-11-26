"""
Unified Data Pipeline
Combines SCADA, images, weather, and text data for RL training
"""

import numpy as np
from typing import Dict, Tuple
import torch

from config import config
from nrel_data_loader import NRELDataLoader
from image_processor import BladeImageProcessor
from weather_api import WeatherDataFetcher
from text_processor import MaintenanceTextProcessor


class DataPipeline:
    """Unified data pipeline for multi-modal RL"""
    
    def __init__(self):
        """Initialize all data sources"""
        print("="* 60)
        print("Initializing Data Pipeline...")
        print("=" * 60)
        
        # Load all data sources
        print("\n[1/4] Loading SCADA data...")
        self.scada_loader = NRELDataLoader()
        
        print("\n[2/4] Loading blade images...")
        self.image_processor = BladeImageProcessor()
        
        print("\n[3/4] Loading weather data...")
        self.weather_fetcher = WeatherDataFetcher()
        
        print("\n[4/4] Loading maintenance logs...")
        self.text_processor = MaintenanceTextProcessor()
        
        print("\n" + "=" * 60)
        print("✓ Data Pipeline initialized successfully!")
        print("=" * 60)
        
        self._print_summary()
    
    def get_multimodal_state(self, turbine_id: int, current_step: int,
                           use_vision: bool = True,
                           use_text: bool = True,
                           use_weather: bool = True) -> Dict[str, np.ndarray]:
        """
        Get complete multimodal state for a turbine
        
        Args:
            turbine_id: Turbine index
            current_step: Current simulation step (in hours)
            use_vision: Include visual features
            use_text: Include text features
            use_weather: Include weather features
        
        Returns:
            Dict with all modality features
        """
        state = {}
        
        # 1. Time series (SCADA) - always included
        state['scada'] = self.scada_loader.get_sequence(turbine_id, current_step)
        
        # 2. Visual features
        if use_vision:
            state['vision'] = self.image_processor.get_aggregated_features(turbine_id)
        else:
            state['vision'] = np.zeros(config.VISION_EMBED_DIM + 2, dtype=np.float32)
        
        # 3. Text features
        if use_text:
            state['text'] = self.text_processor.get_text_features(turbine_id)
        else:
            state['text'] = np.zeros(config.TEXT_EMBED_DIM + 1, dtype=np.float32)
        
        # 4. Weather features
        if use_weather:
            state['weather'] = self.weather_fetcher.get_weather_for_turbine(turbine_id, current_step)
        else:
            state['weather'] = np.zeros(len(config.WEATHER_FEATURES) * 3, dtype=np.float32)
        
        return state
    
    def get_batch_states(self, turbine_ids: list, current_step: int,
                        **kwargs) -> Dict[str, torch.Tensor]:
        """
        Get batched multimodal states for multiple turbines
        
        Args:
            turbine_ids: List of turbine indices
            current_step: Current simulation step
            **kwargs: Additional arguments for get_multimodal_state
        
        Returns:
            Dict with batched tensors for each modality
        """
        batch_states = {
            'scada': [],
            'vision': [],
            'text': [],
            'weather': []
        }
        
        for turbine_id in turbine_ids:
            state = self.get_multimodal_state(turbine_id, current_step, **kwargs)
            for key in batch_states.keys():
                batch_states[key].append(state[key])
        
        # Convert to tensors
        for key in batch_states.keys():
            batch_states[key] = torch.FloatTensor(np.array(batch_states[key]))
        
        return batch_states
    
    def get_state_dimensions(self) -> Dict[str, Tuple[int, ...]]:
        """Get dimensions of each modality"""
        dims = {
            'scada': (config.SCADA_SEQUENCE_LENGTH, len(config.SCADA_FEATURES)),
            'vision': (config.VISION_EMBED_DIM + 2,),  # +2 for defect rate and severity
            'text': (config.TEXT_EMBED_DIM + 1,),  # +1 for severity
            'weather': (len(config.WEATHER_FEATURES) * 3,)  # current + forecast_mean + forecast_std
        }
        return dims
    
    def _print_summary(self):
        """Print summary of loaded data"""
        print("\nData Pipeline Summary:")
        print("-" * 60)
        
        # SCADA stats
        scada_stats = self.scada_loader.get_statistics()
        print(f"✓ SCADA Data:")
        print(f"    Turbines: {scada_stats['num_turbines']}")
        print(f"    Features: {scada_stats['num_features']}")
        print(f"    Sequence Length: {scada_stats['sequence_length']} hours")
        
        # Image stats
        image_stats = self.image_processor.get_statistics()
        print(f"\n✓ Visual Data:")
        print(f"    Total Images: {image_stats['total_images']}")
        print(f"    Feature Dim: {image_stats['feature_dim']}")
        print(f"    Defect Rate: {image_stats['defect_rate']:.2%}")
        
        # Weather stats
        weather_stats = self.weather_fetcher.get_statistics()
        print(f"\n✓ Weather Data:")
        print(f"    Locations: {weather_stats['num_locations']}")
        print(f"    Features: {len(weather_stats['features'])}")
        if 'days_available' in weather_stats:
            print(f"    Days Available: {weather_stats['days_available']:.0f}")
        
        # Text stats
        text_stats = self.text_processor.get_statistics()
        print(f"\n✓ Text Data:")
        print(f"    Total Logs: {text_stats['total_logs']}")
        print(f"    Embedding Dim: {text_stats['embedding_dim']}")
        print(f"    Critical Logs: {text_stats['critical_logs']}")
        
        # State dimensions
        print(f"\n✓ State Dimensions:")
        dims = self.get_state_dimensions()
        for modality, dim in dims.items():
            print(f"    {modality}: {dim}")
        
        print("-" * 60)


# Test the pipeline
if __name__ == "__main__":
    print("Testing Data Pipeline...")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = DataPipeline()
    
    print("\n" + "=" * 60)
    print("Testing single turbine state...")
    print("=" * 60)
    
    # Get state for turbine 0 at step 100
    state = pipeline.get_multimodal_state(turbine_id=0, current_step=100)
    
    print("\nState shapes:")
    for modality, data in state.items():
        print(f"  {modality}: {data.shape}")
    
    print("\n" + "=" * 60)
    print("Testing batched states...")
    print("=" * 60)
    
    # Get batched states
    batch_states = pipeline.get_batch_states(
        turbine_ids=[0, 1, 2, 3, 4],
        current_step=100
    )
    
    print("\nBatch shapes:")
    for modality, tensor in batch_states.items():
        print(f"  {modality}: {tensor.shape}")
    
    print("\n" + "=" * 60)
    print("✓ Data Pipeline test passed!")
    print("=" * 60)
