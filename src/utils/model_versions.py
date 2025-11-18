# version_manager.py
import json
import os
from stable_baselines3 import PPO
import sys, os 
root = os.path.abspath(os.path.join(os.path.dirname(__file__),"../.."))
sys.path.insert(0, root)


class ModelVersionManager:
    def __init__(self, base_path="models/PPO/model_versions"):
        self.base_path = base_path
        self.history_file = os.path.join(base_path, "version_history.json")
        self.version_history = self.load_version_history()
    
    def load_version_history(self):
        """Load version history from JSON file"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}
    
    def save_version_history(self):
        """Save version history to JSON file"""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(self.version_history, f, indent=2)
    
    def get_next_version(self):
        """Get the next version number (v1, v2, etc.)"""
        existing_versions = [v for v in self.version_history.keys() if v.startswith('v')]
        if not existing_versions:
            return 'v1'
        
        # Extract numbers and find max
        version_numbers = []
        for v in existing_versions:
            try:
                num = int(v[1:])  # Remove 'v' and convert to int
                version_numbers.append(num)
            except ValueError:
                continue
        
        next_num = max(version_numbers) + 1 if version_numbers else 1
        return f'v{next_num}'
    
    def save_best_model(self, model, performance_metrics, task_scores, notes=""):
        """Save model as new version with metadata"""
        import datetime
        
        version_id = self.get_next_version()
        version_path = os.path.join(self.base_path, version_id)
        os.makedirs(version_path, exist_ok=True)
        
        # Save model
        model.save(os.path.join(version_path, "model"))
        
        # Create metadata
        metadata = {
            'version': version_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'performance_metrics': performance_metrics,
            'task_scores': task_scores,
            'total_timesteps': model.num_timesteps,
            'notes': notes,
            'model_type': 'PPO'
        }
        
        # Save metadata
        with open(os.path.join(version_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update history
        self.version_history[version_id] = {
            'timestamp': metadata['timestamp'],
            'combined_score': performance_metrics.get('combined_score', 0),
            'best_task': max(task_scores.items(), key=lambda x: x[1])[0] if task_scores else 'unknown'
        }
        self.save_version_history()
        
        print(f"🏆 NEW BEST MODEL: Saved as {version_id}")
        print(f"   Path: {version_path}")
        print(f"   Combined Score: {performance_metrics.get('combined_score', 0):.2f}")
        
        return version_id
    
    def get_best_version(self):
        """Get the version with highest combined score"""
        if not self.version_history:
            return None
        
        best_version = max(self.version_history.items(), 
                          key=lambda x: x[1].get('combined_score', -float('inf')))
        return best_version[0]
    
    def load_best_model(self):
        """Load the best performing model"""
        best_version = self.get_best_version()
        if not best_version:
            return None, None
        
        model_path = os.path.join(self.base_path, best_version, "model.zip")
        if not os.path.exists(model_path):
            return None, None
            
        model = PPO.load(model_path)
        return model, best_version
    
    def list_versions(self):
        """List all saved versions with scores"""
        if not self.version_history:
            print("No versions found.")
            return
        
        print("🏆 SAVED VERSIONS:")
        print("-" * 50)
        for version, info in sorted(self.version_history.items()):
            score = info.get('combined_score', 0)
            best_task = info.get('best_task', 'unknown')
            timestamp = info.get('timestamp', '')[:16]
            print(f"  {version:4} | Score: {score:7.2f} | Best: {best_task:12} | {timestamp}")
    
    def load_version(self, version_id):
        """Load a specific version"""
        model_path = os.path.join(self.base_path, version_id, "model.zip")
        if not os.path.exists(model_path):
            print(f"❌ Version {version_id} not found.")
            return None
        
        model = PPO.load(model_path)
        print(f"✅ Loaded {version_id}")
        return model
    
    def print_version_history(self):
        """Print all saved versions"""
        if not self.version_history:
            print("No versions found in history.")
            return
            
        print("\n📋 VERSION HISTORY:")
        print("-" * 60)
        for version, info in sorted(self.version_history.items()):
            print(f"  {version:4} | Score: {info.get('combined_score', 0):7.2f} | "
                  f"Best: {info.get('best_task', 'unknown'):12} | "
                  f"Time: {info['timestamp'][:19]}")

# Usage example:
if __name__ == "__main__":
    manager = ModelVersionManager()
    manager.list_versions()
    
    # Load the best model
    best_model, version = manager.load_best_model()
    if best_model:
        print(f"🎯 Best model loaded: {version}")