#!/usr/bin/env python3
"""
Train Hybrid Model from Manual Data (Behavioral Cloning)

Trains a neural network policy to imitate human manual flight actions.
Uses supervised learning (behavioral cloning) on state-action pairs.

Workflow:
1. Load all manual flight data files
2. Preprocess: normalize states/actions, split train/val
3. Train neural network: state → action
4. Evaluate on validation set
5. Save trained model for deployment on Tello
"""

import pickle
import numpy as np
from pathlib import Path
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm


class FlightDataset(Dataset):
    """PyTorch dataset for manual flight data"""
    
    def __init__(self, states, actions):
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


class BehaviorCloningPolicy(nn.Module):
    """Neural network policy for imitating manual flight"""
    
    def __init__(self, state_dim=12, action_dim=4, hidden_sizes=[256, 256]):
        super().__init__()
        
        layers = []
        in_dim = state_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_size))
            in_dim = hidden_size
        
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        
        self.state_mean = None
        self.state_std = None
    
    def forward(self, state):
        if self.state_mean is not None:
            state = (state - self.state_mean) / (self.state_std + 1e-8)
        return self.network(state)
    
    def set_normalization(self, mean, std):
        """Set state normalization parameters"""
        self.state_mean = torch.FloatTensor(mean)
        self.state_std = torch.FloatTensor(std)


def load_all_manual_data(data_dir):
    """Load all .pkl files from directory"""
    data_dir = Path(data_dir)
    pkl_files = sorted(data_dir.glob("*.pkl"))
    
    if not pkl_files:
        raise ValueError(f"No .pkl files found in {data_dir}")
    
    all_states = []
    all_actions = []
    
    print(f"\nLoading {len(pkl_files)} flight recordings...")
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        states = data['states']
        actions = data['actions']
        label = data.get('trajectory_label', 'unknown')
        
        all_states.append(states)
        all_actions.append(actions)
        
        print(f"  ✓ {pkl_file.name}: {len(states)} samples ({label})")
    
    states = np.vstack(all_states)
    actions = np.vstack(all_actions)
    
    print(f"\nCombined dataset:")
    print(f"  Total samples: {len(states)}")
    print(f"  State shape: {states.shape}")
    print(f"  Action shape: {actions.shape}")
    
    return states, actions


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for states, actions in dataloader:
        states = states.to(device)
        actions = actions.to(device)
        
        pred_actions = model(states)
        loss = criterion(pred_actions, actions)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for states, actions in dataloader:
            states = states.to(device)
            actions = actions.to(device)
            
            pred_actions = model(states)
            loss = criterion(pred_actions, actions)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def plot_training_curves(train_losses, val_losses, save_path):
    """Plot training and validation loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Behavioral Cloning Training', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curves to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Hybrid model from manual data (Behavioral Cloning)')
    parser.add_argument('--data-dir', type=str, default='data/tello_flights',
                        help='Directory containing manual flight .pkl files')
    parser.add_argument('--output-dir', type=str, default='models/manual_bc',
                        help='Output directory for trained model')
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[256, 256],
                        help='Hidden layer sizes')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("BEHAVIORAL CLONING TRAINING")
    print(f"{'='*60}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Hidden sizes: {args.hidden_sizes}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    
    states, actions = load_all_manual_data(args.data_dir)
    
    print(f"\n{'='*60}")
    print("Data Statistics")
    print(f"{'='*60}")
    print(f"State statistics:")
    print(f"  Mean: {states.mean(axis=0)}")
    print(f"  Std:  {states.std(axis=0)}")
    print(f"\nAction statistics:")
    print(f"  Mean: {actions.mean(axis=0)}")
    print(f"  Std:  {actions.std(axis=0)}")
    print(f"  Range: [{actions.min():.3f}, {actions.max():.3f}]")
    
    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0)
    
    X_train, X_val, y_train, y_val = train_test_split(
        states, actions, test_size=args.val_split, random_state=42, shuffle=True
    )
    
    print(f"\n{'='*60}")
    print("Train/Val Split")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    train_dataset = FlightDataset(X_train, y_train)
    val_dataset = FlightDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    
    model = BehaviorCloningPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=args.hidden_sizes
    )
    model.set_normalization(state_mean, state_std)
    model.to(args.device)
    
    print(f"\n{'='*60}")
    print("Model Architecture")
    print(f"{'='*60}")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    print(f"\n{'='*60}")
    print("Training...")
    print(f"{'='*60}")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss = validate(model, val_loader, criterion, args.device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'state_mean': state_mean,
                'state_std': state_std,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'state_dim': state_dim,
                'action_dim': action_dim,
                'hidden_sizes': args.hidden_sizes,
            }, output_dir / 'best_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Best: {best_val_loss:.6f}")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {output_dir / 'best_model.pth'}")
    
    plot_path = output_dir / 'training_curves.png'
    plot_training_curves(train_losses, val_losses, plot_path)
    
    metadata = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hidden_sizes': args.hidden_sizes,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'best_val_loss': float(best_val_loss),
        'final_train_loss': float(train_losses[-1]),
        'final_val_loss': float(val_losses[-1]),
        'num_train_samples': len(X_train),
        'num_val_samples': len(X_val),
        'state_mean': state_mean.tolist(),
        'state_std': state_std.tolist(),
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"{'='*60}")
    print(f"1. Test trained policy on Tello drone:")
    print(f"   python src/real_drone/run_tello.py --controller hybrid --model {output_dir / 'best_model.pth'}")
    print(f"\n2. Compare with PID performance:")
    print(f"   python src/testing/compare_controllers.py --pid-trajectory <learned_trajectory.pkl> --hybrid-model {output_dir / 'best_model.pth'}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
