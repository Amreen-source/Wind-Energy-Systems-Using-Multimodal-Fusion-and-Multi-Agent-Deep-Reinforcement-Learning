"""
Complete Experiment Pipeline
End-to-end workflow from data loading to results generation
"""

import argparse
import sys
from pathlib import Path
import time

from config import config, validate_config
from train import Trainer
from evaluate import Evaluator


def run_complete_pipeline(args):
    """Run complete training and evaluation pipeline"""
    
    print("\n" + "="*80)
    print(" "*20 + "PREDICTIVE MAINTENANCE PIPELINE")
    print(" "*15 + "Multi-Agent Deep Reinforcement Learning")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    # Step 1: Validate configuration
    print("Step 1/5: Validating configuration...")
    try:
        validate_config()
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
    
    # Step 2: Data verification
    print("\nStep 2/5: Verifying data...")
    if not config.NREL_DATA_PATH.exists():
        print("⚠️  NREL data not found, will use synthetic data")
    else:
        print("✓ NREL data found")
    
    if not config.BLADE_IMAGES_PATH.exists():
        print("⚠️  Blade images not found, will use synthetic features")
    else:
        print("✓ Blade images found")
    
    # Step 3: Training
    if not args.skip_training:
        print("\nStep 3/5: Training models...")
        print("-"*80)
        
        trainer = Trainer()
        
        try:
            if args.resume:
                checkpoint = config.CHECKPOINTS_DIR / 'interrupted_model.pt'
                if checkpoint.exists():
                    print(f"Resuming from {checkpoint}")
                    trainer.load_checkpoint('interrupted_model.pt')
            
            trainer.train(total_timesteps=args.timesteps)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user")
            trainer._save_checkpoint('interrupted_model.pt')
            if not args.force_eval:
                print("Use --resume to continue training later")
                sys.exit(0)
    else:
        print("\nStep 3/5: Skipping training (--skip-training flag)")
    
    # Step 4: Evaluation
    print("\nStep 4/5: Evaluating models...")
    print("-"*80)
    
    model_path = config.CHECKPOINTS_DIR / 'best_model.pt'
    
    if not model_path.exists():
        print(f"⚠️  Best model not found at {model_path}")
        # Try final model
        model_path = config.CHECKPOINTS_DIR / 'final_model.pt'
        if not model_path.exists():
            print("❌ No trained model found. Please train first.")
            sys.exit(1)
    
    evaluator = Evaluator(model_path)
    
    # Run evaluation
    metrics = evaluator.evaluate(num_episodes=args.eval_episodes)
    
    # Compare baselines
    comparison_df = evaluator.compare_baselines()
    
    # Generate visualizations
    evaluator.visualize_results(comparison_df)
    
    # Generate report
    evaluator.generate_report(comparison_df)
    
    # Step 5: Summary
    print("\nStep 5/5: Generating summary...")
    print("-"*80)
    
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    
    print(f"\n{'='*80}")
    print(" "*30 + "PIPELINE COMPLETED!")
    print(f"{'='*80}")
    print(f"\nTotal time: {hours}h {minutes}m")
    print(f"\nResults saved to:")
    print(f"  - Models: {config.CHECKPOINTS_DIR}")
    print(f"  - Figures: {config.FIGURES_DIR}")
    print(f"  - Tables: {config.TABLES_DIR}")
    print(f"  - Logs: {config.LOGS_DIR}")
    print(f"\n{'='*80}\n")


def run_quick_test():
    """Run quick test with minimal settings"""
    
    print("\n" + "="*80)
    print(" "*30 + "QUICK TEST MODE")
    print("="*80 + "\n")
    
    print("Running abbreviated test (100 steps)...")
    
    # Override config for quick test
    config.TOTAL_TIMESTEPS = 100
    config.MAX_STEPS_PER_EPISODE = 10
    config.NUM_EVAL_EPISODES = 2
    config.LOG_FREQUENCY = 10
    
    trainer = Trainer()
    trainer.train(total_timesteps=100)
    
    print("\n✓ Quick test completed successfully!")
    print("System is working. Run full training with: python run_experiment.py --mode full")


def main():
    parser = argparse.ArgumentParser(
        description='Run predictive maintenance experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (100 steps)
  python run_experiment.py --mode test
  
  # Full training and evaluation
  python run_experiment.py --mode full
  
  # Custom training duration
  python run_experiment.py --timesteps 500000
  
  # Evaluation only
  python run_experiment.py --skip-training
  
  # Resume interrupted training
  python run_experiment.py --resume
        """
    )
    
    parser.add_argument('--mode', type=str, default='full',
                       choices=['test', 'full'],
                       help='Run mode: test (quick) or full')
    
    parser.add_argument('--timesteps', type=int, default=config.TOTAL_TIMESTEPS,
                       help=f'Training timesteps (default: {config.TOTAL_TIMESTEPS:,})')
    
    parser.add_argument('--eval-episodes', type=int, default=config.NUM_EVAL_EPISODES,
                       help=f'Evaluation episodes (default: {config.NUM_EVAL_EPISODES})')
    
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training, only evaluate')
    
    parser.add_argument('--resume', action='store_true',
                       help='Resume from interrupted training')
    
    parser.add_argument('--force-eval', action='store_true',
                       help='Force evaluation even if training interrupted')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        run_quick_test()
    else:
        run_complete_pipeline(args)


if __name__ == "__main__":
    main()
