import collections
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np

# Handle both flwr-installed environment and standalone unit testing
try:
    import flwr as fl
    NumPyClientBase = fl.client.NumPyClient
except ImportError:
    class NumPyClientBase:
        pass

from src.models.cifar_cnn import SimpleCNN

class FlowerClient(NumPyClientBase):
    """
    Flower NumPyClient that performs local training and evaluation on an edge device.
    """
    def __init__(self, cid: int, train_loader, test_loader, device: str = "cpu"):
        self.cid = cid
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device(device)
        self.model = SimpleCNN().to(self.device)

    def get_parameters(self, config: Dict[str, str] = None) -> List[np.ndarray]:
        """Extract model weights as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Update local model weights from received NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = collections.OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, any]) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train the model locally.
        Receives parameters from server, trains for 'local_epochs', and returns updated parameters.
        """
        self.set_parameters(parameters)
        self.model.train()

        # Dynamic local epochs (can be dictated by server Battery Governor)
        epochs = int(config.get("local_epochs", 3))
        lr = float(config.get("lr", 0.01))

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

        num_examples = 0
        total_steps = 0

        for _ in range(epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                num_examples += len(labels)
                total_steps += 1

        # Metrics returned to server (including step count for FedNova normalization)
        metrics = {
            "cid": self.cid,
            "local_epochs": epochs,
            "total_steps": total_steps,
        }

        return self.get_parameters(), num_examples, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, any]) -> Tuple[float, int, Dict]:
        """Evaluate current model on local test partition."""
        self.set_parameters(parameters)
        self.model.eval()

        criterion = nn.CrossEntropyLoss()
        correct, total_loss, total_samples = 0, 0.0, 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                total_loss += criterion(outputs, labels).item() * len(labels)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total_samples += len(labels)

        loss = total_loss / max(total_samples, 1)
        accuracy = correct / max(total_samples, 1)

        return float(loss), total_samples, {"accuracy": float(accuracy)}
