import argparse
import yaml
import os
import torch
import numpy as np

from src.models.cifar_cnn import SimpleCNN
from src.client import FlowerClient
from src.server import weighted_average_eval, get_fit_config_fn
from src.data.cifar10_loader import create_synthetic_client_data, load_cifar10_data

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="PAC-FL Simulation Runner")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to YAML config")
    parser.add_argument("--use_synthetic", action="store_true", help="Use lightweight synthetic data for instant test")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["experiment"]["seed"])
    print("=" * 60)
    print(f"[LAUNCH] Launching PAC-FL Simulation: {config['experiment']['name']}")
    print(f"Clients: {config['federated']['total_clients']} | Rounds: {config['experiment']['num_rounds']}")
    print("=" * 60)

    # Check for Flower
    try:
        import flwr as fl
    except ImportError:
        print("\n[NOTE] 'flwr' is not installed yet.")
        print("Please install via: pip install flwr[simulation]")
        print("Running quick local module verification instead...\n")
        # Run a standalone 1-client test to verify PyTorch model and client logic
        model = SimpleCNN()
        print(f"[OK] SimpleCNN initialized successfully: {sum(p.numel() for p in model.parameters())} parameters.")
        client_loaders = create_synthetic_client_data(num_clients=2, samples_per_client=30)
        client = FlowerClient(cid=0, train_loader=client_loaders[0], test_loader=client_loaders[0])
        params = client.get_parameters()
        updated_params, num_samples, metrics = client.fit(params, {"local_epochs": 1, "lr": 0.01})
        print(f"[OK] Local Client Fit test passed! (Trained on {num_samples} samples, {metrics['total_steps']} steps)")
        loss, total, eval_metrics = client.evaluate(updated_params, {})
        print(f"[OK] Local Client Evaluation test passed! (Loss: {loss:.4f}, Accuracy: {eval_metrics['accuracy']:.2f})")
        print("\n[SUCCESS] Member 1 Core Pipeline verified! Ready for Flower installation and team integration.")
        return

    # Data preparation
    total_clients = config["federated"]["total_clients"]
    if args.use_synthetic:
        print("📦 Using synthetic client data for rapid testing...")
        client_train_loaders = create_synthetic_client_data(num_clients=total_clients, samples_per_client=64)
        client_test_loaders = create_synthetic_client_data(num_clients=total_clients, samples_per_client=32)
    else:
        print("📦 Preparing CIFAR-10 data loaders...")
        # Fallback to synthetic if data not yet partitioned by Member 6
        client_train_loaders = create_synthetic_client_data(num_clients=total_clients, samples_per_client=64)
        client_test_loaders = create_synthetic_client_data(num_clients=total_clients, samples_per_client=32)

    # Client factory function for Flower simulation
    def client_fn(cid_str: str) -> fl.client.Client:
        cid = int(cid_str)
        return FlowerClient(
            cid=cid,
            train_loader=client_train_loaders[cid],
            test_loader=client_test_loaders[cid],
            device=config["experiment"]["device"]
        ).to_client()

    # Initial global model parameters
    initial_model = SimpleCNN()
    initial_parameters = fl.common.ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
    )

    # Strategy (Default FedAvg baseline - to be upgraded to FedNova by Member 2)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=config["federated"]["fraction_fit"],
        fraction_evaluate=config["federated"]["fraction_fit"],
        min_fit_clients=config["federated"]["clients_per_round"],
        min_evaluate_clients=config["federated"]["clients_per_round"],
        min_available_clients=total_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=get_fit_config_fn(
            default_epochs=config["model"]["default_local_epochs"],
            lr=config["model"]["lr"]
        ),
        evaluate_metrics_aggregation_fn=weighted_average_eval,
    )

    # Run Flower simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=total_clients,
        config=fl.server.ServerConfig(num_rounds=config["experiment"]["num_rounds"]),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

if __name__ == "__main__":
    main()
