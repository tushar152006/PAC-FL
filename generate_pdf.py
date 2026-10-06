import sys
from fpdf import FPDF

class BlueprintPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(15, 15, 15)

    def header(self):
        if self.page_no() > 1:
            self.set_font("helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 7, "Battery-Efficient & Representation-Fair Federated Learning: Research Blueprint (v2.0)", 0, 0, "L")
            self.cell(0, 7, "Option A - IEEE Track", 0, 1, "R")
            self.set_draw_color(220, 220, 220)
            self.set_line_width(0.3)
            self.line(15, 12, 195, 12)
            self.ln(3)

    def footer(self):
        self.set_y(-14)
        self.set_draw_color(220, 220, 220)
        self.set_line_width(0.3)
        self.line(15, 283, 195, 283)
        self.set_font("helvetica", "", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 8, "Confidential - Research Team Blueprint (3-Month Roadmap / 6 Members)", 0, 0, "L")
        self.cell(0, 8, f"Page {self.page_no()} of {{nb}}", 0, 0, "R")

    def chapter_title(self, num_str, title_str):
        self.ln(3)
        self.set_fill_color(240, 244, 250)
        self.set_draw_color(30, 58, 138)
        self.set_line_width(0.8)
        self.set_font("helvetica", "B", 12)
        self.set_text_color(30, 58, 138)
        self.cell(0, 8, f"  {num_str}. {title_str}", fill=True, border="L", ln=True)
        self.ln(2)

    def sub_title(self, title_str):
        self.ln(1)
        self.set_font("helvetica", "B", 10)
        self.set_text_color(20, 30, 60)
        self.cell(0, 6, title_str, ln=True)
        self.ln(1)

    def body_p(self, text):
        self.set_font("helvetica", "", 9.5)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5, text)
        self.ln(1.5)

    def bullet_p(self, bold_prefix, text):
        self.set_font("helvetica", "B", 9.5)
        self.set_text_color(30, 58, 138)
        self.write(5, "  * " + bold_prefix + " ")
        self.set_font("helvetica", "", 9.5)
        self.set_text_color(40, 40, 40)
        self.write(5, text)
        self.ln(6)

    def code_block(self, lines):
        self.set_fill_color(245, 247, 250)
        self.set_draw_color(210, 220, 230)
        self.set_line_width(0.3)
        self.set_font("courier", "", 7.5)
        self.set_text_color(30, 40, 50)
        
        # Calculate height
        h = len(lines) * 3.8 + 4
        if self.get_y() + h > 270:
            self.add_page()
            
        cur_y = self.get_y()
        self.rect(15, cur_y, 180, h, "DF")
        self.set_y(cur_y + 2)
        for line in lines:
            self.set_x(18)
            self.cell(0, 3.8, line, ln=True)
        self.ln(3)

def generate_pdf():
    pdf = BlueprintPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # Document Header Title
    pdf.set_fill_color(30, 58, 138)
    pdf.rect(15, 15, 180, 2.5, "F")
    pdf.set_y(20)
    pdf.set_font("helvetica", "B", 17)
    pdf.set_text_color(25, 45, 110)
    pdf.multi_cell(0, 7.5, "Battery-Efficient & Representation-Fair Federated Learning\nResearch Blueprint (v2.0)")
    pdf.ln(1)

    pdf.set_font("helvetica", "I", 9.5)
    pdf.set_text_color(80, 90, 105)
    pdf.cell(0, 5, "An Edge-Resource Adaptive Framework with FedNova, INT8 Quantization & Participation Fairness", ln=True)
    pdf.cell(0, 5, "Target Publication Track: Option A (IEEE Access / Workshop / Regional Conference) | 3-Month Plan", ln=True)
    pdf.ln(3)

    # Section 1
    pdf.chapter_title("1", "Executive Summary & Validated Research Scope")
    pdf.body_p(
        "Federated Learning (FL) enables distributed edge devices (smartphones, IoT sensors) to collaboratively "
        "train global machine learning models without transferring raw data. However, real-world edge devices "
        "exhibit severe resource heterogeneity: varied battery levels (15% to 100%), differing network connectivity, "
        "and unpredictable dropouts. Naive greedy battery-aware client selection (e.g., EAFL) creates severe "
        "demographic representation bias under non-IID data distributions, silencing budget devices."
    )
    pdf.body_p(
        "Furthermore, allowing devices to train with varying local epochs without aggregation correction introduces "
        "Objective Inconsistency (Client Drift), causing model divergence (Wang et al., NeurIPS 2020)."
    )
    pdf.body_p("Our upgraded framework reconciles energy efficiency, mathematical convergence, and fairness by coordinating:")
    pdf.bullet_p("Rule-Based Battery Governor:", "Dynamically places devices into Full, Eco, or Sleep tiers per round based on live battery.")
    pdf.bullet_p("FedNova Aggregation Engine:", "Normalizes client updates by local iteration counts, mathematically preventing client drift.")
    pdf.bullet_p("INT8 Uniform Quantization:", "Compresses parameter uploads by 75% (4x data reduction) without high edge CPU sort overhead.")
    pdf.bullet_p("Participation Fairness Credit:", "Tracks client selection counts to ensure chronically low-battery clients are not starved.")

    # Section 2
    pdf.chapter_title("2", "Core Research Questions (RQs)")
    pdf.bullet_p("RQ1 (Energy vs. Accuracy):", "Can adaptive local epochs combined with INT8 quantization reduce total energy consumption by >= 40% with <= 1.5% Top-1 accuracy loss vs. vanilla FedAvg?")
    pdf.bullet_p("RQ2 (Mathematical Convergence):", "Does FedNova eliminate the objective inconsistency and client drift observed when standard FedAvg is paired with variable local epochs?")
    pdf.bullet_p("RQ3 (Communication Efficiency):", "What are the empirical bandwidth savings (MB transferred) of INT8 quantization across communication rounds?")
    pdf.bullet_p("RQ4 (Fairness & Bias Audit):", "Does greedy battery selection cause systematic class accuracy drops on Dirichlet non-IID splits, and does our Fairness Credit restore fair representation (Jain's Fairness Index >= 0.85)?")

    # Section 3
    pdf.chapter_title("3", "System Architecture & Operational Pipeline")
    diagram = [
        "+-----------------------------------------------------------------------------+",
        "|                         CENTRAL SERVER (AGGREGATOR)                         |",
        "|  - Fairness Credit Tracker (Tracks selection history & credit balances)     |",
        "|  - Client Selector (Balances battery health & fairness credits)             |",
        "|  - FedNova Aggregation Engine (Normalizes updates by local step counts)     |",
        "|  - Dequantization Module (Reconstructs FP32 tensors from INT8 packets)      |",
        "+--------------------------------------+--------------------------------------+",
        "                                       | Dispatches Global Model w_t",
        "                                       v",
        "    +----------------------------------+----------------------------------+",
        "    v                                                                     v",
        "+------------------------------------+ +------------------------------------+",
        "|      CLIENT i (HIGH BATTERY)       | |       CLIENT j (LOW BATTERY)       |",
        "|  State: Battery > 60%              | |  State: 30% <= Battery <= 60%      |",
        "|  Assigned Tier: FULL PERFORMANCE   | |  Assigned Tier: ECO MODE           |",
        "+------------------------------------+ +------------------------------------+",
        "|  - Local Epochs: E_i = 5           | |  - Local Epochs: E_j = 2           |",
        "|  - Local Steps: tau_i = 5 * B      | |  - Local Steps: tau_j = 2 * B      |",
        "|  - Precision: FP32                 | |  - Precision: INT8 Quantized (4x)  |",
        "+-----------------+------------------+ +-----------------+------------------+",
        "                  |                                      |",
        "                  +------------------+-------------------+",
        "                                     | Uploads updates (Delta w_i, tau_i)",
        "                                     v",
        "+-----------------------------------------------------------------------------+",
        "|                           FEDNOVA NORMALIZATION                             |",
        "|       Delta w_global = sum p_i * (Delta w_i / tau_i)                        |",
        "|       w_{t+1} = w_t + tau_effective * Delta w_global                        |",
        "+-----------------------------------------------------------------------------+"
    ]
    pdf.code_block(diagram)

    pdf.sub_title("Operational Tier Rules:")
    pdf.bullet_p("Full Performance Tier (Battery > 60%):", "Full local training (E = 5 epochs), standard FP32 precision.")
    pdf.bullet_p("Eco Tier (30% <= Battery <= 60%):", "Reduced training (E = 2 epochs), INT8 quantization applied to model updates.")
    pdf.bullet_p("Sleep Tier (Battery < 30%):", "Preserves device power; participates only when Fairness Credit exceeds threshold.")

    # Section 4
    pdf.chapter_title("4", "Mathematical Foundations")
    pdf.sub_title("4.1 FedNova Normalized Aggregation")
    pdf.body_p(
        "Standard FedAvg computes Delta w = sum p_i * Delta w_i. When clients execute different local epochs (E_i), "
        "local progress tau_i varies, pulling the global model toward high-epoch clients. FedNova normalizes updates "
        "by local progress: d_i = Delta w_i / tau_i. The global update is weighted by effective steps: "
        "Delta w_global = tau_eff * sum(p_i * d_i), preserving objective consistency and ensuring true convergence."
    )
    pdf.sub_title("4.2 INT8 Uniform Symmetric Quantization")
    pdf.body_p(
        "Model update tensors are mapped into signed 8-bit integers: Scale S = max(|Delta w|) / 127. "
        "Quantized weights q = clamp(round(Delta w / S), -127, 127). The client transmits q and scalar S, "
        "reducing transmission bandwidth by 75% in O(M) linear time with minimal edge compute."
    )
    pdf.sub_title("4.3 Energy Consumption Model")
    pdf.body_p(
        "Total round energy per client: E_total = E_comp + E_comm. Compute energy E_comp = E_i * |D_i| * C_model * kappa * f^2. "
        "Communication energy E_comm = P_tx * (Bytes_sent / Data_rate). Battery discharge is decremented accordingly each round."
    )
    pdf.sub_title("4.4 Jain's Fairness Index")
    pdf.body_p(
        "To audit selection equity across all K clients, we evaluate: J(x) = (sum x_i)^2 / (K * sum x_i^2), where x_i is the "
        "cumulative participation count of client i. J in [1/K, 1.0], with 1.0 representing ideal uniform participation."
    )

    # Section 5 - Team Roles
    pdf.chapter_title("5", "Team Roles & Responsibilities (6 Members)")
    pdf.body_p("To ensure balanced workload, equal contribution, and clear authorship sections for the IEEE paper:")

    # Table of roles
    pdf.set_font("helvetica", "B", 8)
    pdf.set_fill_color(30, 58, 138)
    pdf.set_text_color(255, 255, 255)
    pdf.set_draw_color(200, 200, 200)
    pdf.set_line_width(0.3)
    
    col_w = [22, 45, 68, 45]
    headers = ["Member", "Assigned Title", "Core Deliverables & Code", "Paper Section"]
    for i in range(4):
        pdf.cell(col_w[i], 7, headers[i], 1, 0, "C", fill=True)
    pdf.ln(7)

    roles_data = [
        ("Member 1", "Tech Lead & System Architect", "Flower simulation runner, client-server loop, YAML config manager, repo coordination.", "Sec III: System Architecture"),
        ("Member 2", "Aggregation & Optimization Lead", "Custom FedNova Flower Strategy, local step normalization, drift mitigation.", "Sec IV: FedNova Convergence"),
        ("Member 3", "Compression & Quantization Lead", "INT8 symmetric quantization/dequantization, bandwidth measurement profiler.", "Sec V: Quantization & Comm."),
        ("Member 4", "Battery & Energy Modeling Lead", "Energy equations (E_comp + E_comm), battery state simulator, discharge tracker.", "Sec VI: Energy Modeling"),
        ("Member 5", "Adaptive Controller & Fairness Lead", "3-tier rule governor, Fairness Credit accumulator, Jain's Fairness Index auditor.", "Sec VII: Adaptive Governor"),
        ("Member 6", "Benchmarking & Paper Lead", "CIFAR-10 Dirichlet non-IID loader, 4 baseline runs, IEEE LaTeX paper compiler.", "Abstract, Intro, Sec VIII: Results")
    ]

    pdf.set_font("helvetica", "", 7.5)
    pdf.set_text_color(30, 30, 30)
    for row in roles_data:
        pdf.set_fill_color(250, 252, 255) if roles_data.index(row) % 2 == 0 else pdf.set_fill_color(240, 245, 250)
        pdf.cell(col_w[0], 7.5, row[0], 1, 0, "C", fill=True)
        pdf.cell(col_w[1], 7.5, row[1], 1, 0, "L", fill=True)
        pdf.cell(col_w[2], 7.5, row[2], 1, 0, "L", fill=True)
        pdf.cell(col_w[3], 7.5, row[3], 1, 1, "L", fill=True)
    pdf.ln(3)

    # Section 6 - Roadmap
    pdf.chapter_title("6", "3-Month (12-Week) Execution Roadmap")
    pdf.sub_title("Month 1: Foundations & Independent Prototyping (Weeks 1 - 4)")
    pdf.bullet_p("Week 1 (Setup):", "Setup Python 3.11, PyTorch, Flower (flwr). M1 sets up repo skeleton; M6 builds CIFAR-10 CNN.")
    pdf.bullet_p("Week 2 (Data & Energy):", "M6 creates Dirichlet non-IID split (alpha=0.5). M4 builds battery.py simulator. M1 runs vanilla FedAvg.")
    pdf.bullet_p("Week 3 (Algorithms):", "M3 codes quantization.py. M2 prototypes FedNova strategy in Flower. M5 codes rule-based governor.")
    pdf.bullet_p("Week 4 (Unit Verification):", "M5 adds Fairness Credits. M3 validates INT8 accuracy impact (<0.5%). Month 1 Milestone met.")

    pdf.sub_title("Month 2: Integration & Baseline Execution (Weeks 5 - 8)")
    pdf.bullet_p("Week 5 (Core Hookup):", "Hook controller and FedNova into Flower fit_round lifecycle (M1, M2, M5). Add INT8 to client.py (M3).")
    pdf.bullet_p("Week 6 (Baselines 1 & 2):", "Run Baseline 1 (Vanilla FedAvg) & Baseline 2 (Greedy EAFL Selection) for 50 rounds. Log accuracy & energy.")
    pdf.bullet_p("Week 7 (Baselines 3 & Proposed):", "Run Baseline 3 (Heterogeneous FedAvg drift test). Run Proposed PAC-FL (FedNova + INT8 + Fairness).")
    pdf.bullet_p("Week 8 (Data Consolidation):", "Consolidate CSV metrics into Pandas dataframes. Validate accuracy within 1.5% and energy savings >= 40%.")

    pdf.sub_title("Month 3: Ablations, Visualization & Paper Writing (Weeks 9 - 12)")
    pdf.bullet_p("Week 9 (Ablation Runs):", "Run ablation experiments: Proposed without INT8; Proposed without Fairness Credits under heavy non-IID (alpha=0.1).")
    pdf.bullet_p("Week 10 (Publication Plots):", "Generate Pareto Frontier (Accuracy vs. Joules), Participation histograms, and Bandwidth MB charts.")
    pdf.bullet_p("Week 11 (Paper Drafting):", "Compile complete IEEE format paper in LaTeX. Each of the 6 members writes designated sections.")
    pdf.bullet_p("Week 12 (Final Polish):", "Peer review, reference verification, presentation slides, code release on GitHub. Ready for submission.")

    # Section 7 - Experimental Design & Baselines
    pdf.add_page()
    pdf.chapter_title("7", "Experimental Design & Benchmarking Protocol")
    pdf.body_p("All experiments are performed on CIFAR-10 across 20-50 simulated edge clients partitioned via Dirichlet non-IID distributions:")
    pdf.bullet_p("Baseline 1 (Vanilla FedAvg):", "Uniform client selection, fixed E=5, standard FP32 precision. Represents accuracy and energy upper bound.")
    pdf.bullet_p("Baseline 2 (Greedy EAFL):", "Selects only top-battery clients (>=50%), fixed E=5. Demonstrates demographic starvation on non-IID data.")
    pdf.bullet_p("Baseline 3 (Unnormalized FedAvg):", "Variable local epochs (E_i) based on battery, aggregated via standard FedAvg. Proves client drift failure.")
    pdf.bullet_p("Proposed Framework (PAC-FL):", "Adaptive local epochs (E_i) + INT8 quantization + FedNova aggregation + Participation Fairness Credits.")

    pdf.sub_title("Target Success Metrics:")
    pdf.bullet_p("Top-1 Accuracy:", "Final accuracy within <= 1.5% of Vanilla FedAvg upper bound.")
    pdf.bullet_p("Energy Reduction:", "At least 40% reduction in aggregate Joules consumed across clients.")
    pdf.bullet_p("Bandwidth Reduction:", "At least 70% reduction in total megabytes uploaded across rounds.")
    pdf.bullet_p("Jain's Fairness Index:", "J >= 0.85 (compared to J <= 0.45 under greedy battery selection).")

    # Section 8 - References
    pdf.chapter_title("8", "Key References for IEEE Publication")
    refs = [
        "1. Beutel, D. J. et al. (2020). Flower: A Friendly Federated Learning Framework. arXiv:2007.14390.",
        "2. Wang, J. et al. (2020). Tackling Objective Inconsistency in Heterogeneous Federated Optimization (FedNova). NeurIPS 33.",
        "3. Arouj, A. et al. (2022). Towards Energy-Aware Federated Learning on Battery-Powered Clients (EAFL). ACM FedEdge.",
        "4. Savoia, M. et al. (2024). Eco-FL: Enhancing Federated Learning Sustainability. Computer Communications, 225.",
        "5. Dettmers, T. et al. (2022). 8-bit Optimizers via Block-wise Dynamic Quantization. ICLR.",
        "6. Marnissi, O. et al. (2025). FairEnergy: Contribution-Based Fairness meets Energy Efficiency in FL. arXiv:2511.15454.",
        "7. Li, T. et al. (2020). Federated Optimization in Heterogeneous Networks (FedProx). MLSys.",
        "8. Jain, R. et al. (1984). A Quantitative Measure of Fairness for Resource Allocation. DEC Tech Report."
    ]
    for ref in refs:
        pdf.set_font("helvetica", "", 8.5)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(0, 4.8, ref, ln=True)

    pdf.output("FL_Battery_Efficiency_Research_Blueprint.md.pdf")
    print("Successfully generated FL_Battery_Efficiency_Research_Blueprint.md.pdf")

if __name__ == "__main__":
    generate_pdf()
