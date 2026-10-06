import os
import sys
from fpdf import FPDF
from fpdf.enums import XPos, YPos

class TeamGuidePDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(15, 15, 15)

    def header(self):
        if self.page_no() > 1:
            self.set_font("helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 7, "PAC-FL: Battery-Efficient & Fair Federated Learning - Team Onboarding Guide", 
                      new_x=XPos.RIGHT, new_y=YPos.TOP)
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
        self.cell(0, 8, "Team Confidential - Project Handbook & Execution Guide (Team of 6)", 
                  new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.cell(0, 8, f"Page {self.page_no()} of {{nb}}", 
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")

    def section_header(self, num_str, title_str):
        self.ln(3)
        self.set_fill_color(238, 242, 255)  # Light indigo/blue
        self.set_draw_color(49, 46, 129)    # Deep indigo
        self.set_line_width(1.0)
        self.set_font("helvetica", "B", 12)
        self.set_text_color(49, 46, 129)
        self.cell(0, 8.5, f"  {num_str}. {title_str}", fill=True, border="L", 
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2.5)

    def sub_header(self, title_str):
        self.ln(1)
        self.set_font("helvetica", "B", 10.5)
        self.set_text_color(30, 41, 59)
        self.cell(0, 6, title_str, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def body_p(self, text):
        self.set_font("helvetica", "", 9.5)
        self.set_text_color(51, 65, 85)
        self.multi_cell(0, 5.2, text)
        self.ln(1.5)

    def highlight_box(self, title, text, bg_rgb=(241, 245, 249), border_rgb=(148, 163, 184)):
        cur_y = self.get_y()
        self.set_font("helvetica", "B", 9.5)
        # Check space
        self.set_fill_color(*bg_rgb)
        self.set_draw_color(*border_rgb)
        self.set_line_width(0.4)
        
        # Estimate height
        lines_count = len(text) // 85 + 3
        box_h = lines_count * 5 + 6
        if cur_y + box_h > 270:
            self.add_page()
            cur_y = self.get_y()
            
        self.rect(15, cur_y, 180, box_h, "DF")
        self.set_xy(18, cur_y + 2.5)
        self.set_text_color(30, 41, 59)
        self.cell(0, 5, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_xy(18, cur_y + 7.5)
        self.set_font("helvetica", "", 9)
        self.set_text_color(71, 85, 105)
        self.multi_cell(174, 4.8, text)
        self.set_y(cur_y + box_h + 3)

    def bullet(self, bold_prefix, text):
        self.set_font("helvetica", "B", 9.5)
        self.set_text_color(49, 46, 129)
        self.write(5.2, "  * " + bold_prefix + " ")
        self.set_font("helvetica", "", 9.5)
        self.set_text_color(51, 65, 85)
        self.write(5.2, text)
        self.ln(5.8)

    def code_box(self, lines):
        self.set_fill_color(248, 250, 252)
        self.set_draw_color(203, 213, 225)
        self.set_line_width(0.3)
        self.set_font("courier", "", 7.5)
        self.set_text_color(30, 41, 59)
        
        h = len(lines) * 3.8 + 4
        if self.get_y() + h > 270:
            self.add_page()
            
        cur_y = self.get_y()
        self.rect(15, cur_y, 180, h, "DF")
        self.set_y(cur_y + 2)
        for line in lines:
            self.set_x(18)
            self.cell(0, 3.8, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)

def create_guide_pdf():
    pdf = TeamGuidePDF()
    pdf.alias_nb_pages()
    
    # ------------------ PAGE 1: TITLE & BIG PICTURE ------------------
    pdf.add_page()
    
    # Top banner
    pdf.set_fill_color(49, 46, 129) # Navy
    pdf.rect(15, 15, 180, 2.5, "F")
    pdf.set_y(20)
    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(30, 27, 75)
    pdf.multi_cell(0, 7.5, "PAC-FL: Battery-Efficient & Fair Federated Learning\nComplete Team Onboarding & Project Handbook")
    pdf.ln(1)
    
    pdf.set_font("helvetica", "I", 9.5)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, "Everything you need to understand: Concept, Architecture, Math, Roles & Month 1 Tasks", 
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 5, "Target Outcome: B.Tech Capstone Project + IEEE Conference / Journal Paper (Team of 6)", 
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    pdf.section_header("1", "The Big Picture: What is Federated Learning (FL)?")
    pdf.body_p(
        "In traditional Machine Learning, all user data (photos, text, sensor logs) is uploaded to a central cloud server "
        "like Google or AWS, where a giant model is trained. However, this raises massive privacy concerns and legal issues "
        "(GDPR, HIPAA)."
    )
    pdf.highlight_box(
        "Real-World Analogy (Google Gboard)",
        "When your phone predicts the next word you type on your keyboard, Google doesn't send your private chat messages "
        "to their servers. Instead, your phone downloads the current model, trains it locally on your typing habits, "
        "and sends back only mathematical parameter updates (gradients). The server merges updates from millions of phones. "
        "This is Federated Learning!"
    )
    
    pdf.section_header("2", "The Real-World Problems We Are Solving")
    pdf.body_p("While Federated Learning sounds great in theory, in reality on battery-powered mobile devices, three huge problems occur:")
    
    pdf.bullet("Problem 1: Rapid Battery Drain & Mid-Round Dropouts.",
               "Training deep neural networks on edge processors drains phone batteries rapidly. Devices with 15-25% battery frequently shut down or disconnect midway through a training round, wasting communication and time.")
               
    pdf.bullet("Problem 2: The 'Unfairness' Trap (Demographic Starvation).",
               "Prior research tried to fix battery drain by simply ignoring low-battery phones ('Greedy Selection'). But in real life, data is Non-IID (people have completely different data). If you only train on high-battery phones, you only train on people with new expensive phones who leave them plugged in! The global model will fail on budget devices.")
               
    pdf.bullet("Problem 3: The Mathematical Breakdown (Client Drift).",
               "If low-battery phones do fewer training epochs (e.g. 1 epoch) while high-battery phones do 5 epochs, standard FedAvg averages them incorrectly. The global model drifts toward high-battery clients and fails to converge.")

    # ------------------ PAGE 2: OUR SOLUTION (PAC-FL) ------------------
    pdf.add_page()
    pdf.section_header("3", "Our Solution: The PAC-FL Framework Explained")
    pdf.body_p(
        "We are building PAC-FL (Privacy-Preserving Adaptive Controller for Federated Learning). "
        "It coordinates 4 smart engineering mechanisms to solve all three problems simultaneously:"
    )

    pdf.sub_header("The 4 Core Pillars of PAC-FL:")
    pdf.bullet("1. Smart 3-Tier Battery Governor:", 
               "Every round, each device evaluates its battery: >60% -> Full Tier (5 epochs); 30%-60% -> Eco Tier (2 epochs); <30% -> Sleep Tier (skip to save battery).")
               
    pdf.bullet("2. INT8 Parameter Quantization (4x Compression):", 
               "Phones in Eco Mode compress their weights from 32-bit floats to 8-bit integers before transmitting. This reduces network payload by 75%, slashing wireless radio transmission energy dramatically.")
               
    pdf.bullet("3. FedNova Normalized Aggregation (Math Fix):", 
               "Instead of naive FedAvg, the server uses FedNova. It divides each device's update by the exact number of training steps it completed, completely eliminating client drift and guaranteeing true mathematical convergence.")
               
    pdf.bullet("4. Participation Fairness Credits (Equity Guarantee):", 
               "The server tracks how often each client participates. If a low-battery phone is skipped multiple rounds, it earns 'Fairness Credits'. When credits build up, it is granted an emergency slot so its unique data is never lost.")

    pdf.sub_header("System Flow Diagram (How a Round Works):")
    diagram = [
        "                      +------------------------------------------+",
        "                      |         CENTRAL SERVER (FLOWER)          |",
        "                      | - Tracks Fairness Credits                |",
        "                      | - Dispatches Global Model Weights w_t    |",
        "                      +--------------------+---------------------+",
        "                                           |",
        "                 +-------------------------+-------------------------+",
        "                 v                                                   v",
        "     +-----------------------+                           +-----------------------+",
        "     |  CLIENT A (90% BATT)  |                           |  CLIENT B (35% BATT)  |",
        "     |  Tier: FULL MODE      |                           |  Tier: ECO MODE       |",
        "     +-----------------------+                           +-----------------------+",
        "     | * Local Epochs: E = 5 |                           | * Local Epochs: E = 2 |",
        "     | * Precision: FP32     |                           | * Precision: INT8 (4x)|",
        "     | * Standard Energy     |                           | * 70% Energy Saved    |",
        "     +-----------+-----------+                           +-----------+-----------+",
        "                 |                                                   |",
        "                 +-------------------------+-------------------------+",
        "                                           v",
        "                      +------------------------------------------+",
        "                      |           FEDNOVA AGGREGATION            |",
        "                      | - Dequantizes INT8 packets to FP32       |",
        "                      | - Normalizes updates by local step count |",
        "                      | - Computes unbiased global model w_{t+1} |",
        "                      +------------------------------------------+"
    ]
    pdf.code_box(diagram)

    # ------------------ PAGE 3: TEAM ROLES & DELIVERABLES ------------------
    pdf.add_page()
    pdf.section_header("4", "Team Roles & Individual Ownership (Team of 6)")
    pdf.body_p(
        "Each member has a clearly defined responsibility, a dedicated code module to build, and a specific section "
        "to author in our final IEEE conference/journal paper. No overlapping confusion!"
    )

    roles = [
        ("Member 1: Tech Lead & FL Architect", 
         "Main Job: Set up the project repository, configure Python virtual environments, and build the Flower (flwr) simulation engine that coordinates clients and server.\n"
         "Code Files: src/server.py, src/client.py, run_simulation.py\n"
         "Paper Section: Section III - System Architecture & Framework Implementation."),

        ("Member 2: Aggregation & Optimization Specialist (FedNova)", 
         "Main Job: Implement the custom Flower Strategy that replaces standard FedAvg with FedNova. Mathematically normalizes updates so varying epochs don't break training.\n"
         "Code Files: src/strategy/fednova_strategy.py\n"
         "Paper Section: Section IV - Mathematical Formulation & Objective Consistency."),

        ("Member 3: Compression & Quantization Engineer", 
         "Main Job: Build the INT8 tensor compression functions (FP32 <-> INT8) in PyTorch. Measure exact megabytes saved per round and verify that accuracy drop is negligible (<0.5%).\n"
         "Code Files: src/compression/quantizer.py\n"
         "Paper Section: Section V - Communication Compression & Quantization Overhead."),

        ("Member 4: Energy & Battery Modeling Specialist", 
         "Main Job: Formulate the physical energy model for compute (FLOPs x clock speed) and wireless transmission. Build the battery state simulator that decrements battery per epoch and upload.\n"
         "Code Files: src/energy/battery_simulator.py, src/energy/energy_model.py\n"
         "Paper Section: Section VI - Edge Energy Modeling & Battery Discharge Simulation."),

        ("Member 5: Adaptive Controller & Fairness Lead", 
         "Main Job: Build the 3-tier battery governor logic (Full/Eco/Sleep) and implement the Fairness Credit counter. Calculate Jain's Fairness Index to prove our method treats devices equitably.\n"
         "Code Files: src/controller/battery_governor.py, src/fairness/fairness_tracker.py\n"
         "Paper Section: Section VII - Adaptive Resource Governor & Representation Fairness."),

        ("Member 6: Data Pipeline, Benchmarks & Paper Lead", 
         "Main Job: Download CIFAR-10, implement the Dirichlet non-IID partitioner (alpha=0.5), coordinate running the 4 experimental baselines, and manage the IEEE Overleaf LaTeX paper template.\n"
         "Code Files: src/data/cifar10_partitioner.py, benchmarks/run_baselines.py\n"
         "Paper Section: Abstract, Section I (Intro), Section VIII (Experimental Evaluation).")
    ]

    for title, desc in roles:
        pdf.set_font("helvetica", "B", 9.5)
        pdf.set_text_color(49, 46, 129)
        pdf.cell(0, 5, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("helvetica", "", 8.5)
        pdf.set_text_color(51, 65, 85)
        pdf.multi_cell(0, 4.4, desc)
        pdf.ln(2)

    # ------------------ PAGE 4: MONTH 1 ACTION PLAN & GLOSSARY ------------------
    pdf.add_page()
    pdf.section_header("5", "Month 1 (Weeks 1 to 4): Step-by-Step Action Plan")
    pdf.body_p("During Month 1, every member works independently on their module. You do NOT need to wait for others!")

    pdf.sub_header("Week 1: Environment & Foundations (Oct 7 - Oct 14)")
    pdf.bullet("All Members:", "Install Python 3.11, PyTorch, Torchvision, and Flower (`pip install flwr[simulation]`). Clone repo.")
    pdf.bullet("Member 1 & Member 6:", "Initialize folder structure and build simple 3-layer CNN on CIFAR-10.")

    pdf.sub_header("Week 2: Data Split & Battery Simulator (Oct 15 - Oct 21)")
    pdf.bullet("Member 6:", "Implement Dirichlet partitioner (`alpha = 0.5`). Verify that client 0 has different class distributions than client 1.")
    pdf.bullet("Member 4:", "Write `battery_simulator.py`. Assign initial batteries (15% to 100%) to 20 clients.")
    pdf.bullet("Member 1:", "Run basic vanilla FedAvg on CIFAR-10 using Flower simulation.")

    pdf.sub_header("Week 3: Algorithm Prototypes (Oct 22 - Oct 28)")
    pdf.bullet("Member 3:", "Write `quantize_tensor(t)` and `dequantize_tensor(q, scale)` in PyTorch. Verify output matches original tensor.")
    pdf.bullet("Member 2:", "Write `FedNovaStrategy` skeleton in Flower that accepts client epoch counts.")
    pdf.bullet("Member 5:", "Write rule conditions: if battery > 60% return Full, if 30-60% return Eco, else Sleep.")

    pdf.sub_header("Week 4: Unit Testing & Milestone 1 (Oct 29 - Nov 5)")
    pdf.bullet("Member 5:", "Add Fairness Credit counter. If client is skipped 3 times, grant credit.")
    pdf.bullet("All Members:", "Run standalone tests for each module. Milestone 1 Complete: All 6 pieces ready for integration in Month 2!")

    pdf.section_header("6", "Quick Glossary: Plain-English Definitions")
    glossary = [
        ("Non-IID Data:", "Non-Independent and Identically Distributed. Means each user's phone has different data (e.g., Client A only has pictures of dogs, Client B only has pictures of planes)."),
        ("Local Epoch (E):", "How many times a phone passes through its own local data before sending updates back."),
        ("INT8 Quantization:", "Compressing 32-bit floating point numbers into 8-bit integers. Reduces file size by 75%."),
        ("Client Drift:", "When different clients train for different amounts of time, the global model gets pulled toward the faster clients and drifts away from the correct answer."),
        ("FedNova:", "A special mathematical algorithm created by researchers at NeurIPS that fixes Client Drift by normalizing update steps."),
        ("Jain's Fairness Index:", "A mathematical score between 0 and 1.0 that measures fairness. 1.0 means every phone participated equally.")
    ]
    for term, definition in glossary:
        pdf.set_font("helvetica", "B", 8.5)
        pdf.set_text_color(30, 41, 59)
        pdf.write(4.5, "  * " + term + " ")
        pdf.set_font("helvetica", "", 8.5)
        pdf.set_text_color(71, 85, 105)
        pdf.write(4.5, definition)
        pdf.ln(5)

    pdf.ln(2)
    pdf.highlight_box(
        "Need Help or Clarification?",
        "Don't worry if you don't know everything on Day 1! We are working as a team. Member 1 (Tech Lead) and your teammates "
        "are here to help. Reach out in the WhatsApp group whenever you get stuck!",
        bg_rgb=(238, 242, 255), border_rgb=(99, 102, 241)
    )

    out_file = "PAC_FL_Complete_Team_Project_Guide.pdf"
    pdf.output(out_file)
    print(f"Successfully generated {out_file}")

if __name__ == "__main__":
    create_guide_pdf()
