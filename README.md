# PAC-FL: Battery-Efficient & Representation-Fair Federated Learning

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Framework: Flower](https://img.shields.io/badge/FL_Framework-Flower_1.8+-orange.svg)](https://flower.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)

An edge-resource adaptive Federated Learning framework that coordinates **adaptive local epochs**, **INT8 symmetric quantization**, **FedNova normalized aggregation**, and **participation fairness credits** on Non-IID CIFAR-10.

---

## 📖 Key Documentation
* **Team Onboarding Guide:** [`PAC_FL_Complete_Team_Project_Guide.pdf`](./PAC_FL_Complete_Team_Project_Guide.pdf)
* **Research Blueprint (v2.0):** [`FL_Battery_Efficiency_Research_Blueprint.md.pdf`](./FL_Battery_Efficiency_Research_Blueprint.md.pdf)

---

## 👥 Team Roles & Responsibilities

| Member | Role | Core Responsibility | Module Path |
| :--- | :--- | :--- | :--- |
| **Member 1** | **Tech Lead & FL Architect** | Flower simulation runner, server/client lifecycle, repo manager. | `src/client.py`, `run_simulation.py` |
| **Member 2** | **Optimization & Aggregation Lead** | Custom FedNova aggregation strategy, step normalization. | `src/strategy/fednova.py` |
| **Member 3** | **Compression & Quantization Engineer** | INT8 symmetric quantization and bandwidth profiler. | `src/compression/quantizer.py` |
| **Member 4** | **Energy & Battery Modeling Specialist** | Compute/comm energy models and battery discharge tracker. | `src/energy/battery_sim.py` |
| **Member 5** | **Adaptive Controller & Fairness Lead** | 3-tier battery governor and Fairness Credit tracker. | `src/controller/governor.py` |
| **Member 6** | **Data Pipeline & Benchmarking Lead** | CIFAR-10 Dirichlet non-IID loader, baseline experiments, IEEE paper. | `src/data/cifar10_loader.py` |

---

## 🛠️ Repository Structure

```text
EPICS/
├── config/
│   └── default_config.yaml         # Experiment configuration
├── src/
│   ├── models/
│   │   └── cifar_cnn.py            # CNN architecture
│   ├── data/
│   │   └── cifar10_loader.py       # Dirichlet Non-IID loader
│   ├── strategy/
│   │   └── fednova.py              # Custom FedNova Strategy
│   ├── compression/
│   │   └── quantizer.py            # INT8 Quantization
│   ├── energy/
│   │   └── battery_sim.py          # Battery discharge simulator
│   ├── controller/
│   │   └── governor.py             # 3-Tier Battery Governor
│   ├── client.py                   # Flower Client implementation
│   └── server.py                   # Flower Server & strategy setup
├── run_simulation.py               # Main simulation entrypoint
├── requirements.txt                # Dependencies
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Clone & Set Up Virtual Environment
```bash
git clone <your-github-repo-url>
cd EPICS

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Baseline Simulation
```bash
python run_simulation.py --config config/default_config.yaml
```

---

## 📅 3-Month Roadmap Summary
* **Month 1 (Weeks 1–4):** Independent module prototyping and unit verification.
* **Month 2 (Weeks 5–8):** Full Flower lifecycle integration & comparative baseline benchmarking.
* **Month 3 (Weeks 9–12):** Ablation experiments, Pareto frontier visualization, and IEEE paper writing.
