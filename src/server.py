from typing import List, Tuple, Dict
import numpy as np

def weighted_average_eval(metrics: List[Tuple[int, Dict]]) -> Dict:
    """
    Aggregate evaluation accuracy across participating clients using sample weights.
    """
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics if "accuracy" in m]
    examples = [num_examples for num_examples, m in metrics if "accuracy" in m]

    if not examples or sum(examples) == 0:
        return {"accuracy": 0.0}

    return {"accuracy": sum(accuracies) / sum(examples)}

def get_fit_config_fn(default_epochs: int = 3, lr: float = 0.01):
    """Factory to generate fit configuration per round."""
    def fit_config(server_round: int) -> Dict:
        return {
            "server_round": server_round,
            "local_epochs": default_epochs,
            "lr": lr,
        }
    return fit_config
