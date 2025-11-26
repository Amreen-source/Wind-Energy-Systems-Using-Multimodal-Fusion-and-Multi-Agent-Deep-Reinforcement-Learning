"""
Configuration file for Renewable Energy Predictive Maintenance
Contains all hyperparameters, paths, and settings
"""

import os
from pathlib import Path

class Config:
    """Main configuration class"""
    
    # ============================================================================
    # PATHS
    # ============================================================================
    ROOT_DIR = Path(__file__).parent
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = ROOT_DIR / "models"
    RESULTS_DIR = ROOT_DIR / "results"
    LOGS_DIR = ROOT_DIR / "logs"
    
    # Dataset paths
    NREL_DATA_PATH = RAW_DATA_DIR / "NREL_WIND_DATA"
    BLADE_IMAGES_PATH = RAW_DATA_DIR / "Wind_Turbine_Blade_Defect_Detection" / "Images"
    BLADE_LABELS_PATH = RAW_DATA_DIR / "Wind_Turbine_Blade_Defect_Detection" / "Label"
    
    # Output paths
    FIGURES_DIR = RESULTS_DIR / "figures"
    TABLES_DIR = RESULTS_DIR / "tables"
    CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
    
    # Create directories if they don't exist
    for dir_path in [PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR,
                     FIGURES_DIR, TABLES_DIR, CHECKPOINTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # ============================================================================
    # ENVIRONMENT SETTINGS
    # ============================================================================
    
    # Wind farm configuration
    NUM_TURBINES = 20  # Number of wind turbines in the farm
    NUM_COMPONENTS_PER_TURBINE = 5  # Major components: blade, gearbox, generator, tower, nacelle
    TOTAL_AGENTS = NUM_TURBINES  # One agent per turbine
    
    # Simulation settings
    SIMULATION_DAYS = 365  # Simulate one year
    HOURS_PER_STEP = 24  # Each step = 1 day
    MAX_STEPS_PER_EPISODE = SIMULATION_DAYS
    
    # Component failure rates (failures per year)
    FAILURE_RATES = {
        'blade': 0.15,
        'gearbox': 0.25,
        'generator': 0.20,
        'tower': 0.05,
        'nacelle': 0.12
    }
    
    # Maintenance costs (in $1000s)
    MAINTENANCE_COSTS = {
        'preventive': {'blade': 5, 'gearbox': 15, 'generator': 10, 'tower': 8, 'nacelle': 12},
        'corrective': {'blade': 25, 'gearbox': 75, 'generator': 50, 'tower': 40, 'nacelle': 60}
    }
    
    # Downtime (in hours)
    DOWNTIME = {
        'preventive': {'blade': 8, 'gearbox': 24, 'generator': 16, 'tower': 12, 'nacelle': 20},
        'corrective': {'blade': 48, 'gearbox': 120, 'generator': 72, 'tower': 60, 'nacelle': 96}
    }
    
    # Energy production
    TURBINE_CAPACITY_MW = 2.5  # MW per turbine
    ENERGY_PRICE_PER_MWH = 50  # $/MWh
    
    # ============================================================================
    # DATA PROCESSING
    # ============================================================================
    
    # SCADA data settings
    SCADA_FEATURES = [
        'wind_speed', 'wind_direction', 'temperature', 'humidity',
        'power_output', 'rotor_speed', 'blade_pitch',
        'gearbox_temp', 'generator_temp', 'nacelle_temp',
        'vibration_x', 'vibration_y', 'vibration_z'
    ]
    SCADA_SEQUENCE_LENGTH = 168  # 1 week of hourly data
    SCADA_SAMPLING_RATE = 1  # hours
    
    # Image processing
    IMAGE_SIZE = 224  # For Vision Transformer
    IMAGE_AUGMENTATION = True
    MAX_IMAGES_PER_TURBINE = 10  # Use up to 10 recent inspection images
    
    # Text processing
    MAX_TEXT_LENGTH = 512  # Max tokens for maintenance logs
    TEXT_MODEL_NAME = "bert-base-uncased"  # or "distilbert-base-uncased" for faster
    
    # Weather API
    WEATHER_FEATURES = ['temp', 'pressure', 'humidity', 'wind_speed', 'wind_deg', 'clouds']
    WEATHER_FORECAST_DAYS = 7
    
    # ============================================================================
    # MODEL ARCHITECTURE
    # ============================================================================
    
    # Vision Transformer
    # VISION_MODEL_NAME = "google/vit-base-patch16-224"  # or "facebook/dinov2-base"
    VISION_MODEL_NAME = "facebook/dinov2-base"
    VISION_EMBED_DIM = 768
    VISION_FREEZE_BACKBONE = False
    
    # Text Encoder
    TEXT_EMBED_DIM = 768
    TEXT_FREEZE_BACKBONE = False
    
    # Time Series Encoder (Temporal Convolutional Network)
    TCN_CHANNELS = [64, 128, 256]
    TCN_KERNEL_SIZE = 3
    TCN_DROPOUT = 0.2
    TIME_SERIES_EMBED_DIM = 256
    
    # Graph Neural Network
    GNN_HIDDEN_DIM = 128
    GNN_NUM_LAYERS = 3
    GNN_DROPOUT = 0.2
    GNN_EMBED_DIM = 128
    
    # Fusion Module
    FUSION_TYPE = "cross_attention"  # Options: "concat", "cross_attention", "multimodal_transformer"
    FUSION_HIDDEN_DIM = 512
    FUSION_NUM_HEADS = 8
    FUSION_DROPOUT = 0.1
    FINAL_STATE_DIM = 512  # Dimension of fused state representation
    
    # ============================================================================
    # REINFORCEMENT LEARNING
    # ============================================================================
    
    # Action space
    # Discrete actions: [no_action, inspect_only, preventive_maintenance, defer_to_next_week]
    NUM_DISCRETE_ACTIONS = 4
    
    # For continuous actions (maintenance intensity, resource allocation)
    NUM_CONTINUOUS_ACTIONS = 2
    ACTION_SPACE_TYPE = "discrete"  # Options: "discrete", "continuous", "hybrid"
    
    # Reward function weights
    REWARD_WEIGHTS = {
        'maintenance_cost': -1.0,
        'energy_loss': -2.0,
        'failure_penalty': -5.0,
        'availability_bonus': 1.0,
        'early_detection_bonus': 0.5
    }
    
    # Multi-agent RL algorithm
    MARL_ALGORITHM = "QMIX"  # Options: "QMIX", "MAPPO", "IQL" (Independent Q-Learning)
    
    # Q-Network architecture
    Q_HIDDEN_DIMS = [512, 256, 128]
    
    # Mixing network (for QMIX)
    MIXING_EMBED_DIM = 64
    MIXING_HYPERNET_HIDDEN = 128
    
    # Communication
    USE_COMMUNICATION = True
    COMM_DIM = 64
    COMM_ROUNDS = 2  # Number of communication rounds per step
    
    # ============================================================================
    # TRAINING HYPERPARAMETERS
    # ============================================================================
    
    # General
    SEED = 42
    DEVICE = "cuda"  # Will auto-switch to "cpu" if CUDA unavailable
    NUM_WORKERS = 4  # For data loading
    
    # Training
    TOTAL_TIMESTEPS = 10000  # Total training timesteps
    BATCH_SIZE = 64
    BUFFER_SIZE = 100_000  # Replay buffer size
    LEARNING_RATE = 3e-4
    GAMMA = 0.99  # Discount factor
    TAU = 0.005  # Soft update coefficient for target network
    
    # Exploration
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 0.995
    EXPLORATION_STEPS = 50_000  # Steps for epsilon decay
    
    # Network updates
    TARGET_UPDATE_FREQUENCY = 1000  # Update target network every N steps
    GRADIENT_STEPS = 1  # Gradient updates per environment step
    MAX_GRAD_NORM = 10.0  # Gradient clipping
    
    # Prioritized Experience Replay
    USE_PER = True
    PER_ALPHA = 0.6
    PER_BETA_START = 0.4
    PER_BETA_END = 1.0
    PER_EPS = 1e-6
    
    # Curriculum learning
    USE_CURRICULUM = True
    CURRICULUM_STAGES = [
        {'max_steps': 100, 'failure_rate_multiplier': 0.3},  # Easy
        {'max_steps': 200, 'failure_rate_multiplier': 0.6},  # Medium
        {'max_steps': 365, 'failure_rate_multiplier': 1.0}   # Full
    ]
    
    # ============================================================================
    # EVALUATION
    # ============================================================================
    
    # Evaluation frequency
    EVAL_FREQUENCY = 10_000  # Evaluate every N training steps
    NUM_EVAL_EPISODES = 10
    
    # Baseline methods for comparison
    BASELINES = [
        'reactive',           # Fix only when broken
        'fixed_schedule',     # Preventive maintenance every N days
        'condition_based',    # Threshold-based maintenance
        'single_agent_dqn',   # Centralized DQN
        'independent_learners', # No cooperation
        'lstm_predictor',     # LSTM-based failure prediction
        'qmix_no_multimodal', # QMIX without vision/text
        'mappo'              # Multi-agent PPO
    ]
    
    # Metrics to track
    METRICS = [
        'total_reward',
        'maintenance_cost',
        'energy_production',
        'system_availability',
        'mtbf',  # Mean Time Between Failures
        'num_failures',
        'num_preventive_maintenance',
        'avg_component_health'
    ]
    
    # ============================================================================
    # LOGGING AND CHECKPOINTING
    # ============================================================================
    
    # TensorBoard
    USE_TENSORBOARD = True
    TENSORBOARD_LOG_DIR = LOGS_DIR / "tensorboard"
    
    # Checkpointing
    SAVE_FREQUENCY = 50_000  # Save checkpoint every N steps
    KEEP_N_CHECKPOINTS = 5  # Keep only N most recent checkpoints
    
    # Logging
    LOG_FREQUENCY = 1000  # Log to console every N steps
    VERBOSE = True
    
    # ============================================================================
    # VISUALIZATION
    # ============================================================================
    
    # Figure settings
    FIGURE_DPI = 300
    FIGURE_FORMAT = 'png'  # 'png', 'pdf', 'svg'
    FIGURE_SIZE = (10, 6)
    
    # Plot style
    PLOT_STYLE = 'seaborn-v0_8-darkgrid'
    COLOR_PALETTE = 'husl'
    
    # Attention visualization
    SAVE_ATTENTION_MAPS = True
    ATTENTION_VIS_FREQUENCY = 5000  # Save attention visualizations every N steps
    
    # ============================================================================
    # ABLATION STUDY CONFIGURATIONS
    # ============================================================================
    
    ABLATION_CONFIGS = {
        'full_model': {
            'use_vision': True,
            'use_text': True,
            'use_timeseries': True,
            'use_graph': True,
            'use_communication': True
        },
        'no_vision': {
            'use_vision': False,
            'use_text': True,
            'use_timeseries': True,
            'use_graph': True,
            'use_communication': True
        },
        'no_text': {
            'use_vision': True,
            'use_text': False,
            'use_timeseries': True,
            'use_graph': True,
            'use_communication': True
        },
        'no_multimodal': {
            'use_vision': False,
            'use_text': False,
            'use_timeseries': True,
            'use_graph': True,
            'use_communication': True
        },
        'no_communication': {
            'use_vision': True,
            'use_text': True,
            'use_timeseries': True,
            'use_graph': True,
            'use_communication': False
        },
        'no_graph': {
            'use_vision': True,
            'use_text': True,
            'use_timeseries': True,
            'use_graph': False,
            'use_communication': True
        }
    }
    
    # ============================================================================
    # API KEYS (Loaded from .env)
    # ============================================================================
    
    @staticmethod
    def load_env():
        """Load environment variables from .env file"""
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv('OPENWEATHER_API_KEY', None)


# Create a global config instance
config = Config()

# Validate configuration
def validate_config():
    """Validate configuration settings"""
    assert config.NUM_TURBINES > 0, "Number of turbines must be positive"
    assert config.BATCH_SIZE > 0, "Batch size must be positive"
    assert config.LEARNING_RATE > 0, "Learning rate must be positive"
    assert 0 <= config.GAMMA <= 1, "Gamma must be in [0, 1]"
    assert config.NREL_DATA_PATH.exists(), f"NREL data not found at {config.NREL_DATA_PATH}"
    assert config.BLADE_IMAGES_PATH.exists(), f"Blade images not found at {config.BLADE_IMAGES_PATH}"
    
    api_key = config.load_env()
    if api_key is None or api_key == "your_api_key_here":
        print("⚠️  WARNING: OpenWeather API key not configured in .env file")
    
    print("✓ Configuration validated successfully")

if __name__ == "__main__":
    validate_config()
    print(f"\nConfiguration Summary:")
    print(f"  Turbines: {config.NUM_TURBINES}")
    print(f"  Total Agents: {config.TOTAL_AGENTS}")
    print(f"  Training Steps: {config.TOTAL_TIMESTEPS:,}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Device: {config.DEVICE}")
    print(f"  MARL Algorithm: {config.MARL_ALGORITHM}")
