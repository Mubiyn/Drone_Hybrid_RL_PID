#!/usr/bin/env python3
import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.training.ppo_trainer import PPOTrainer


def main():
    parser = argparse.ArgumentParser(description='Train PPO on drone task')
    parser.add_argument('--task', type=str, required=True,
                       choices=['hover', 'hover_extended', 'waypoint_delivery', 
                               'figure8', 'circle', 'emergency_landing'],
                       help='Task to train on')
    parser.add_argument('--timesteps', type=int, default=1_000_000,
                       help='Total training timesteps')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--no-domain-rand', action='store_true',
                       help='Disable domain randomization')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate existing model')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model for evaluation')
    
    args = parser.parse_args()
    
    # Configure trainer (merge with defaults)
    config = {
        'learning_rate': args.learning_rate,
        'total_timesteps': args.timesteps,
        'domain_randomization': not args.no_domain_rand,
    }
    # PPOTrainer will merge this with default config
    
    trainer = PPOTrainer(args.task, config=config)
    
    if args.eval_only:
        # Evaluation mode
        if args.model_path is None:
            model_path = f"models/ppo/{args.task}/ppo_{args.task}_final.zip"
        else:
            model_path = args.model_path
        
        print(f"Loading model from: {model_path}")
        trainer.create_env()
        trainer.load_model(model_path)
        trainer.evaluate(n_episodes=20)
    else:
        # Training mode
        print(f"\n{'='*60}")
        print(f"PPO Training Configuration:")
        print(f"  Task: {args.task}")
        print(f"  Timesteps: {args.timesteps:,}")
        print(f"  Learning Rate: {args.learning_rate}")
        print(f"  Domain Randomization: {not args.no_domain_rand}")
        print(f"{'='*60}\n")
        
        trainer.train()
        
        # Quick evaluation
        print("\nRunning post-training evaluation...")
        trainer.evaluate(n_episodes=10)


if __name__ == '__main__':
    main()
