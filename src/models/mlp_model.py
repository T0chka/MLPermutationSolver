import torch
import torch.nn as nn
from .base_model import BaseModel

class MLPNetwork(nn.Module):
    """Neural network architecture for permutation solving"""
    
    def __init__(self, input_size: int, hidden_sizes: list = [512, 256, 128]):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, size),
                nn.BatchNorm1d(size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = size
        
        # Output layer - single value regression
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)

class MLPModel(BaseModel):
    """MLP implementation for permutation solving"""
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list = [128],
        learning_rate: float = 0.001,
        batch_size: int = 1024,
        epochs: int = 50
    ):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Initialize network
        self.model = MLPNetwork(input_size, hidden_sizes)
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def train(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Train the model on the given data using mini-batch training"""
        # Convert input to float32
        X = X.float()
        y = y.float()
        
        self.model.train()
        self.model = self.model.to(X.device)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        for epoch in range(self.epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in dataloader:
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            if epoch % 10 == 0:  # Print every 10 epochs
                print(f"Epoch {epoch}/{self.epochs}, Loss: {avg_loss:.6f}")
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict distances for given states"""
        # Convert input to float32
        X = X.float()
        
        self.model.eval()
        self.model = self.model.to(X.device)
        
        with torch.no_grad():
            predictions = self.model(X)
        
        return predictions
    
    def save(self, path: str) -> None:
        """Save model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs
        }, path)
    
    def load(self, path: str) -> None:
        """Load model from disk"""
        checkpoint = torch.load(path)
        
        # Recreate model architecture
        self.input_size = checkpoint['input_size']
        self.hidden_sizes = checkpoint['hidden_sizes']
        self.learning_rate = checkpoint['learning_rate']
        self.batch_size = checkpoint['batch_size']
        self.epochs = checkpoint['epochs']
        
        self.model = MLPNetwork(self.input_size, self.hidden_sizes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Load state dicts
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 