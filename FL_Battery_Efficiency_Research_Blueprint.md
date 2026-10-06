# Battery-Efficient & Representation-Fair Federated Learning: Research Blueprint (v2.0)
**Target Publication:** IEEE Access / IEEE Workshop / Regional Conference (Option A Track)  
**Framework:** Flower (`flwr`) + PyTorch  
**Benchmark:** CIFAR-10 with Dirichlet Non-IID Partitioning  

---

## 1. Executive Summary & Problem Formulation

Federated Learning (FL) allows distributed edge devices (smartphones, IoT sensors) to collaboratively train a global machine learning model without sharing raw data. However, real-world edge devices suffer from **extreme resource heterogeneity**:
1. Devices operate on battery with vastly different remaining charge (from <15% to 100%).
2. Low-battery devices frequently disconnect or drop out mid-round, causing straggler latency and wasted communication.
3. If an FL scheduler naively selects only high-battery clients (as in prior works like EAFL), it creates severe **representation bias** on non-IID data distributions—chronically under-powering and silencing lower-tier user segments.
4. If clients perform differing amounts of local training ($E_i$ epochs) to conserve power, standard FedAvg suffers from **Objective Inconsistency (Client Drift)**, causing the global model to diverge.

### The Upgraded Core Contribution
We propose a practical, mathematically sound, energy-aware Federated Learning framework that jointly coordinates:
* **Rule-Based Adaptive Battery Governor:** Dynamically classifies devices into *Full*, *Eco*, or *Sleep* tiers per round based on live battery levels.
* **FedNova Normalized Aggregation:** Replaces naive FedAvg to mathematically eliminate client drift caused by heterogeneous local epoch counts ($E_i$).
* **INT8 Symmetric Quantization:** Compresses upstream model updates by $75\%$ ($4\times$ data reduction), slashing transmission energy without incurring heavy edge CPU sorting overhead.
* **Participation Fairness Credit:** Tracks historical device selection and assigns fairness credits to prevent demographic starvation on non-IID datasets.

---

## 2. Core Research Questions (RQs)

* **RQ1 (Energy vs. Accuracy):** Can the joint controller (adaptive epochs + INT8 quantization) reduce total system energy consumption by $\ge 40\%$ while keeping final Top-1 accuracy degradation within $\le 1.5\%$ compared to vanilla FedAvg?
* **RQ2 (Mathematical Convergence):** Does FedNova aggregation eliminate the objective inconsistency (client drift) observed when naive FedAvg is combined with heterogeneous local epochs?
* **RQ3 (Communication Efficiency):** How does INT8 quantization impact convergence rate and cumulative bandwidth consumption (MB transferred) across communication rounds?
* **RQ4 (Fairness & Bias Audit):** Does greedy battery-aware client selection systematically degrade class accuracy on non-IID splits, and does our Participation Fairness Credit mechanism restore fair representation (measured by Jain's Fairness Index)?

---

## 3. System Architecture & Operational Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CENTRAL SERVER (AGGREGATOR)                         │
│  - Fairness Credit Tracker (Tracks selection history & credit balances)     │
│  - Client Selector (Balances battery health & fairness credits)             │
│  - FedNova Aggregation Engine (Normalizes updates by local step counts)     │
│  - Dequantization Module (Reconstructs FP32 tensors from INT8 packets)      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Broadcasts Global Model w_t
                                       ▼
    ┌──────────────────────────────────┴──────────────────────────────────┐
    ▼                                                                     ▼
┌────────────────────────────────────┐ ┌────────────────────────────────────┐
│      CLIENT i (HIGH BATTERY)       │ │       CLIENT j (LOW BATTERY)       │
│  State: Battery > 60%              │ │  State: 30% <= Battery <= 60%      │
│  Assigned Tier: FULL PERFORMANCE   │ │  Assigned Tier: ECO MODE           │
├────────────────────────────────────┤ ├────────────────────────────────────┤
│  - Local Epochs: E_i = 5           │ │  - Local Epochs: E_j = 2           │
│  - Local Steps: \tau_i = 5 * B     │ │  - Local Steps: \tau_j = 2 * B     │
│  - Compression: None (FP32)        │ │  - Compression: INT8 Quantized     │
│  - Energy Draw: Standard           │ │  - Energy Draw: Minimized          │
└─────────────────┬──────────────────┘ └─────────────────┬──────────────────┘
                  │                                      │
                  └──────────────────┬───────────────────┘
                                     │ Uploads updates (\Delta w_i, \tau_i)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FEDNOVA NORMALIZATION                             │
│       \Delta w_{global} = \sum p_i \cdot (\Delta w_i / \tau_i)              │
│       w_{t+1} = w_t + \tau_{effective} \cdot \Delta w_{global}              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Operational Tier Definitions
1. **Full Performance Tier ($\text{Battery} > 60\%$):** Client performs full training ($E = 5$ epochs) and transmits standard precision FP32 parameters.
2. **Eco Tier ($30\% \le \text{Battery} \le 60\%$):** Client performs reduced training ($E = 2$ epochs) and applies INT8 quantization to compress updates by $4\times$.
3. **Sleep Tier ($\text{Battery} < 30\%$):** Client sleeps to preserve power, unless its **Fairness Credit** exceeds threshold $\theta_{\text{fair}}$, granting an emergency lightweight slot ($E = 1$, INT8).

---

## 4. Mathematical Formulations

### 4.1. FedNova Normalized Aggregation
When clients perform varying numbers of local iterations $\tau_i = E_i \times (\text{local steps})$, standard FedAvg calculates an unnormalized weighted average:
$$\Delta w_{\text{FedAvg}} = \sum_{i \in S_t} \frac{n_i}{N} \Delta w_i$$
This introduces client drift toward clients with larger $\tau_i$. FedNova corrects this by normalizing local updates by their step count:
$$d_i = \frac{\Delta w_i}{\tau_i}$$
$$\tau_{\text{eff}} = \sum_{i \in S_t} p_i \tau_i, \quad \Delta w_{\text{global}} = \tau_{\text{eff}} \sum_{i \in S_t} p_i d_i$$
$$w_{t+1} = w_t - \eta \Delta w_{\text{global}}$$
This preserves objective consistency and guarantees unbiased global convergence.

### 4.2. INT8 Uniform Symmetric Quantization
To compress gradient/weight updates $\Delta w$ from 32-bit floats to 8-bit signed integers:
$$S = \frac{\max(|\Delta w|)}{127}$$
$$q = \text{clamp}\left(\left\lfloor \frac{\Delta w}{S} \right\rceil, -127, 127\right)$$
$$\Delta \tilde{w} = q \times S$$
* Transmitted data size is reduced from 32 bits to 8 bits per parameter ($75\%$ reduction).
* Unlike Top-K sorting, uniform quantization executes in $O(M)$ linear time with virtually zero edge CPU overhead.

### 4.3. Energy Consumption Model
Total device energy consumed in round $t$ is formulated as:
$$E_{\text{total}, i} = E_{\text{comp}, i} + E_{\text{comm}, i}$$
* **Computation Energy:** $E_{\text{comp}, i} = E_i \cdot |D_i| \cdot C_{\text{model}} \cdot \kappa \cdot f^2$ (where $C_{\text{model}}$ is FLOPs/sample, $\kappa$ is hardware capacitance, $f$ is CPU clock frequency).
* **Communication Energy:** $E_{\text{comm}, i} = P_{\text{tx}} \cdot \frac{B(\psi_i)}{R_{\text{link}}}$ (where $B(\psi_i)$ is payload bytes transmitted under compression ratio $\psi_i$, $R_{\text{link}}$ is transmission data rate).

### 4.4. Jain's Fairness Index
To quantify whether client selection is equitable across all $K$ clients:
$$\mathcal{J}(x) = \frac{\left( \sum_{i=1}^{K} x_i \right)^2}{K \sum_{i=1}^{K} x_i^2}$$
where $x_i$ is the cumulative number of times client $i$ was selected. $\mathcal{J} \in [1/K, 1]$, where $1.0$ represents completely fair, uniform selection.

---

## 5. Team Structure & Roles (6-Member Team)

| Member | Title | Core Responsibilities | Final Paper Section |
| :--- | :--- | :--- | :--- |
| **Member 1** | **Tech Lead & System Architect** | Flower simulation framework setup, core client-server lifecycle, config management, repository lead. | Section III: System Architecture & Implementation |
| **Member 2** | **Aggregation & Optimization Lead** | Implementation of custom Flower `FedNovaStrategy`, client drift analysis, aggregation math validation. | Section IV: FedNova Convergence & Objective Consistency |
| **Member 3** | **Compression & Quantization Engineer** | INT8 symmetric quantization and dequantization pipeline, FP32 baseline, bandwidth profiler. | Section V: Quantization & Communication Reduction |
| **Member 4** | **Battery & Energy Modeling Specialist** | Formulate energy consumption equations ($E_{\text{comp}} + E_{\text{comm}}$), battery discharge state tracker. | Section VI: Edge Energy & Battery Discharge Modeling |
| **Member 5** | **Adaptive Controller & Fairness Lead** | Rule-based 3-tier governor, Fairness Credit accumulator, Jain's Fairness Index calculation. | Section VII: Adaptive Governor & Representation Fairness |
| **Member 6** | **Benchmarking, Data Pipeline & Paper Lead** | CIFAR-10 Dirichlet non-IID loader ($\alpha=0.5, 0.1$), comparative baseline runs, IEEE paper compilation. | Abstract, Section I (Intro), Section VIII (Results) |

---

## 6. 3-Month (12-Week) Execution Roadmap

```
Month 1: Weeks 1–4  ──► Modular Build & Independent Prototypes
Month 2: Weeks 5–8  ──► Full Integration & Comparative Baselines
Month 3: Weeks 9–12 ──► Ablation Studies, Result Plots & IEEE Paper Draft
```

### Month 1: Foundations & Independent Modules (Weeks 1–4)
* **Week 1:** Setup Python 3.11 virtual environment, PyTorch, Torchvision, and Flower (`flwr[simulation]`). Member 1 initializes repo structure; Member 6 builds basic CNN on CIFAR-10.
* **Week 2:** Member 6 implements Dirichlet non-IID partitioner ($\alpha = 0.5$). Member 4 implements `battery.py` discharge simulator. Member 1 executes vanilla FedAvg in Flower.
* **Week 3:** Member 3 develops `quantization.py` (PyTorch tensor $\leftrightarrow$ INT8 uint8). Member 2 prototypes `FedNovaStrategy` in Flower. Member 5 codes rule-based threshold governor.
* **Week 4:** Member 5 adds Fairness Credit accumulator and Jain's Index script. Member 3 validates that INT8 incurs $<0.5\%$ standalone accuracy drop. **Milestone: All standalone modules pass unit tests.**

### Month 2: Integration & Baseline Execution (Weeks 5–8)
* **Week 5:** Integrate all modules into Flower's `fit_round` cycle (`client.py`, `server.py`, `strategy.py`).
* **Week 6:** Run **Baseline 1 (Vanilla FedAvg)** and **Baseline 2 (Greedy EAFL Selection)** on 50 communication rounds. Log accuracy, energy, and client dropouts.
* **Week 7:** Run **Baseline 3 (Varying $E_i$ on naive FedAvg)** to capture client drift. Run **Proposed System (FedNova + INT8 + Battery Governor + Fairness Credits)**.
* **Week 8:** Consolidate experimental metrics into Pandas tables. Confirm accuracy delta $\le 1.5\%$ while energy savings $\ge 40\%$. **Milestone: Full baseline suite completed.**

### Month 3: Ablations, Visualization & Paper Writing (Weeks 9–12)
* **Week 9:** Run ablation studies: (a) Proposed without INT8 quantization; (b) Proposed without Fairness Credits under heavy non-IID skew ($\alpha = 0.1$).
* **Week 10:** Generate publication-quality plots:
  * Accuracy vs. Total Joules (Pareto Frontier).
  * Client Selection Distribution bar charts (Fairness vs. Greedy).
  * Cumulative Network Bandwidth (MB) over rounds.
* **Week 11:** Draft IEEE format paper in Overleaf/LaTeX (all 6 members write their designated sections).
* **Week 12:** Peer review, proofreading, citation cleanup, and project presentation/defense preparation.

---

## 7. Comparative Baselines & Success Criteria

### 7.1. Baselines for Experimental Evaluation
1. **Baseline 1: Vanilla FedAvg:** Uniform client selection, fixed $E = 5$, FP32 transmission. Represents accuracy upper-bound and energy upper-bound.
2. **Baseline 2: Greedy Battery Selection (EAFL-style):** Only selects top-battery clients ($\ge 50\%$), fixed $E = 5$. Demonstrates representation starvation on non-IID splits.
3. **Baseline 3: Unnormalized Heterogeneous FedAvg:** Variable $E_i$ based on battery, but aggregated via standard FedAvg. Proves client drift and mathematical inconsistency.
4. **Proposed Framework (PAC-FL):** Adaptive $(E_i, \text{INT8}) + \text{FedNova} + \text{Fairness Credits}$.

### 7.2. Quantitative Success Criteria
* **Accuracy Cost:** Final Top-1 accuracy within **$\le 1.5\%$** of Vanilla FedAvg.
* **Energy Reduction:** At least **$40\%$ reduction** in aggregate Joules consumed across clients.
* **Communication Reduction:** At least **$70\%$ reduction** in total MB transmitted.
* **Fairness Guarantee:** Jain's Fairness Index $\mathcal{J} \ge 0.85$ (compared to $\le 0.45$ under greedy selection).

---

## 8. Key References for IEEE Paper

1. **Beutel, D. J. et al. (2020)**. Flower: A Friendly Federated Learning Framework. *arXiv:2007.14390*.
2. **Wang, J. et al. (2020)**. Tackling Objective Inconsistency in Heterogeneous Federated Optimization (FedNova). *Advances in Neural Information Processing Systems (NeurIPS 33)*.
3. **Arouj, A. et al. (2022)**. Towards Energy-Aware Federated Learning on Battery-Powered Clients (EAFL). *ACM FedEdge*.
4. **Savoia, M. et al. (2024)**. Eco-FL: Enhancing Federated Learning Sustainability. *Computer Communications, 225*.
5. **Dettmers, T. et al. (2022)**. 8-bit Optimizers via Block-wise Dynamic Quantization. *ICLR*.
6. **Marnissi, O. et al. (2025)**. FairEnergy: Contribution-Based Fairness meets Energy Efficiency in Federated Learning. *arXiv:2511.15454*.
7. **Li, T. et al. (2020)**. Federated Optimization in Heterogeneous Networks (FedProx). *MLSys*.
8. **Jain, R. et al. (1984)**. A Quantitative Measure of Fairness and Discrimination for Resource Allocation in Shared Systems. *DEC Tech Report*.
