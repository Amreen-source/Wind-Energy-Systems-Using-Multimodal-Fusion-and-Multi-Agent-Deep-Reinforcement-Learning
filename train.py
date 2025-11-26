"""
Training Pipeline for Multi-Agent Predictive Maintenance
Complete training loop with logging and checkpointing
"""

import torch
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import json
from torch.utils.tensorboard import SummaryWriter

from config import config
from wind_farm_env import WindFarmEnv
from marl_agents import MARLAgent
from replay_buffer import ReplayBuffer


class Trainer:
    """Trainer for MARL agents"""
    
    def __init__(self, save_dir: str = None):
        """
        Initialize trainer
        
        Args:
            save_dir: Directory to save checkpoints and logs
        """
        self.save_dir = Path(save_dir) if save_dir else config.CHECKPOINTS_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize environment
        print("Initializing environment...")
        self.env = WindFarmEnv()
        
        # Initialize agent
        print("Initializing MARL agent...")
        self.agent = MARLAgent(
            num_agents=config.NUM_TURBINES,
            state_dim=config.FINAL_STATE_DIM,
            action_dim=config.NUM_DISCRETE_ACTIONS
        )
        
        # Initialize replay buffer
        print("Initializing replay buffer...")
        self.buffer = ReplayBuffer()
        
        # TensorBoard
        if config.USE_TENSORBOARD:
            self.writer = SummaryWriter(config.TENSORBOARD_LOG_DIR)
        else:
            self.writer = None
        
        # Training stats
        self.total_steps = 0
        self.episode_count = 0
        self.best_reward = -float('inf')
        
        print("✓ Trainer initialized")
    
    def train(self, total_timesteps: int = None):
        """
        Main training loop
        
        Args:
            total_timesteps: Total training timesteps
        """
        total_timesteps = total_timesteps or config.TOTAL_TIMESTEPS
        
        print(f"\n{'='*60}")
        print(f"Starting training for {total_timesteps:,} timesteps")
        print(f"{'='*60}\n")
        
        pbar = tqdm(total=total_timesteps, desc="Training")
        
        while self.total_steps < total_timesteps:
            # Run episode
            episode_reward, episode_steps = self._run_episode()
            
            self.episode_count += 1
            self.total_steps += episode_steps
            pbar.update(episode_steps)
            
            # Training
            if len(self.buffer) > config.BATCH_SIZE:
                for _ in range(config.GRADIENT_STEPS):
                    loss = self._training_step()
                    
                    if self.writer:
                        self.writer.add_scalar('Loss/train', loss, self.total_steps)
            
            # Logging
            if self.episode_count % config.LOG_FREQUENCY == 0:
                self._log_progress(episode_reward)
            
            # Evaluation
            if self.total_steps % config.EVAL_FREQUENCY == 0:
                eval_reward = self._evaluate()
                
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    self._save_checkpoint('best_model.pt')
            
            # Save checkpoint
            if self.total_steps % config.SAVE_FREQUENCY == 0:
                self._save_checkpoint(f'checkpoint_{self.total_steps}.pt')
            
            # Update target networks
            if self.total_steps % config.TARGET_UPDATE_FREQUENCY == 0:
                self.agent.update_target_networks()
            
            # Decay epsilon
            self.agent.decay_epsilon()
        
        pbar.close()
        
        # Final save
        self._save_checkpoint('final_model.pt')
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Total episodes: {self.episode_count}")
        print(f"Best reward: {self.best_reward:.2f}")
        print(f"{'='*60}\n")
        
        if self.writer:
            self.writer.close()
    
    def _run_episode(self) -> tuple:
        """Run one episode and collect experiences"""
        states, _ = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            # Select actions
            actions = self.agent.select_actions(states, explore=True)
            
            # Environment step
            next_states, rewards, terminated, truncated, info = self.env.step(actions)
            done = terminated or truncated
            
            # Store experience
            self.buffer.push(states, actions, rewards, next_states, done)
            
            # Update for next iteration
            states = next_states
            episode_reward += sum(rewards.values())
            episode_steps += 1
            
            if episode_steps >= config.MAX_STEPS_PER_EPISODE:
                break
        
        return episode_reward, episode_steps
    
    def _training_step(self) -> float:
        """Single training step"""
        # Sample batch
        batch, indices, weights = self.buffer.sample(config.BATCH_SIZE)
        
        # Train agent
        loss = self.agent.train_step(batch)
        
        # Update priorities (if using PER)
        if config.USE_PER:
            # Compute TD errors for priority update (simplified)
            priorities = np.abs(np.random.randn(len(indices))) + config.PER_EPS
            self.buffer.update_priorities(indices, priorities)
        
        return loss
    
    def _evaluate(self, num_episodes: int = None) -> float:
        """Evaluate agent performance"""
        num_episodes = num_episodes or config.NUM_EVAL_EPISODES
        
        eval_rewards = []
        
        for _ in range(num_episodes):
            states, _ = self.env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < config.MAX_STEPS_PER_EPISODE:
                actions = self.agent.select_actions(states, explore=False)
                next_states, rewards, terminated, truncated, _ = self.env.step(actions)
                
                episode_reward += sum(rewards.values())
                states = next_states
                done = terminated or truncated
                steps += 1
            
            eval_rewards.append(episode_reward)
        
        avg_reward = np.mean(eval_rewards)
        
        if self.writer:
            self.writer.add_scalar('Reward/eval', avg_reward, self.total_steps)
        
        if config.VERBOSE:
            print(f"\nEvaluation: Avg reward = {avg_reward:.2f} "
                  f"(std = {np.std(eval_rewards):.2f})")
        
        return avg_reward
    
    def _log_progress(self, episode_reward: float):
        """Log training progress"""
        if config.VERBOSE:
            print(f"\nEpisode {self.episode_count} | "
                  f"Steps: {self.total_steps:,} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Epsilon: {self.agent.epsilon:.3f}")
        
        if self.writer:
            self.writer.add_scalar('Reward/train', episode_reward, self.total_steps)
            self.writer.add_scalar('Epsilon', self.agent.epsilon, self.total_steps)
    
    def _save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        checkpoint_path = self.save_dir / filename
        
        self.agent.save(str(checkpoint_path))
        
        # Save training state
        state_path = self.save_dir / f'trainer_state_{filename}'
        torch.save({
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'best_reward': self.best_reward
        }, state_path)
        
        if config.VERBOSE and 'best' in filename:
            print(f"✓ Saved best model to {checkpoint_path}")
    
    def load_checkpoint(self, filename: str):
        """Load training checkpoint"""
        checkpoint_path = self.save_dir / filename
        
        if checkpoint_path.exists():
            self.agent.load(str(checkpoint_path))
            
            # Load training state
            state_path = self.save_dir / f'trainer_state_{filename}'
            if state_path.exists():
                state = torch.load(state_path)
                self.total_steps = state['total_steps']
                self.episode_count = state['episode_count']
                self.best_reward = state['best_reward']
            
            print(f"✓ Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")


def train_baseline(baseline_name: str):
    """Train baseline method for comparison"""
    print(f"\nTraining baseline: {baseline_name}")
    
    # Simplified baseline training (placeholder)
    # In full implementation, would have separate logic for each baseline
    
    if baseline_name == 'reactive':
        print("  Reactive maintenance: No training needed (rule-based)")
    elif baseline_name == 'fixed_schedule':
        print("  Fixed schedule: No training needed (rule-based)")
    else:
        print(f"  Training {baseline_name}...")
        # Would implement specific baseline training here
    
    print(f"✓ Baseline {baseline_name} ready")


def main():
    """Main training function"""
    print("\n" + "="*60)
    print("Multi-Agent Predictive Maintenance Training")
    print("="*60 + "\n")
    
    # Set random seeds
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # Train main model
    trainer = Trainer()
    
    try:
        trainer.train(total_timesteps=config.TOTAL_TIMESTEPS)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        trainer._save_checkpoint('interrupted_model.pt')
    
    # Train baselines (optional)
    if config.VERBOSE:
        print("\n" + "="*60)
        print("Training Baselines")
        print("="*60)
        
        for baseline in config.BASELINES[:3]:  # Train first 3 baselines
            train_baseline(baseline)
    
    print("\n" + "="*60)
    print("Training pipeline completed!")
    print(f"Models saved to: {config.CHECKPOINTS_DIR}")
    print(f"Logs saved to: {config.LOGS_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
