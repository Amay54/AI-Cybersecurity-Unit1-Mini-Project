# AI-Driven Cyber Threat Awareness and Detection Prototype
**Unit 1 Academic Mini-Project: AI in Cybersecurity**  
**Author:** Dhruv Arora  
**Repository:** [DhruvArora664/ai-cybersecurity-unit1-mini-project-dhruv](https://github.com/DhruvArora664/ai-cybersecurity-unit1-mini-project-dhruv)

An end-to-end academic cybersecurity machine learning project that processes telemetry from the **TON_IoT Network Dataset**, performs exploratory threat data analysis, trains and benchmarks Machine Learning (**DecisionTreeClassifier**) and Deep Learning (**MLPClassifier**) detection engines, and evaluates operational security trade-offs (False Positives vs. False Negatives) under aggressive threat mitigation postures.

---

## 📁 Project Directory Structure

```text
ai-cybersecurity-unit1-mini-project-dhruv/
├── data/
│   └── .gitkeep                     # Preserves cached data directory (gitignored)
├── src/
│   ├── main.ipynb                   # Fully executed, interactive Jupyter Notebook with live outputs
│   └── main.py                      # Standalone, end-to-end Python pipeline script
├── confusion_matrix_simulation.png  # High-res visualization: Standard vs. Aggressive Posture
├── eda_plots.png                    # High-res visualization: Traffic and Feature Distribution
├── .gitignore                       # Git ignore rules for venv, data caches, and checkpoints
├── requirements.txt                 # Pinned dependencies (numpy, pandas, scikit-learn, etc.)
└── README.md                        # Project documentation and benchmark report
```

---

## 📊 Dataset Overview: TON_IoT Network Dataset

- **Origin**: Australian Defence Force Academy (ADFA) / UNSW Canberra Cyber & Kaggle (`arnobbhowmik/ton-iot-network-dataset`).
- **Telemetry**: Real-world IoT and network flow telemetry collected using the Zeek (Bro) network analysis framework.
- **Labels**:
  - `label`: Binary classification target (`0` = Normal traffic, `1` = Suspicious / Attack traffic).
  - `type`: Attack category breakdown (e.g., `ddos`, `dos`, `injection`, `backdoor`, `mitm`, `normal`).

---

## 🔬 Benchmark Results & Comparative Evaluation

Models were evaluated on a stratified test set of **20,000 network flows** (80/20 train-test split on 100,000 telemetry records) with strict isolation to prevent data leakage (encoders and scalers fitted solely on the training partition).

### Model Performance Comparison

| Metric | Decision Tree (ML) | MLP Neural Network (DL) |
| :--- | :---: | :---: |
| **Accuracy** | **99.89%** | 99.42% |
| **Precision** | **0.9993** | 0.9950 |
| **Recall** | **0.9994** | 0.9978 |
| **F1-Score** | **0.9993** | 0.9964 |
| **Training Time** | **0.731s** | 14.436s (~20x slower) |
| **Interpretability** | High (White-box rules) | Low (Black-box weights) |
| **Edge Deployment** | Ideal for line-rate edge IDS | Requires hardware acceleration |

---

## 🛡️ Operational Threat Posture & Ethical Simulation (Task 5)

In operational Security Operations Centers (SOCs), threat detection thresholds are often tuned to prioritize zero-day mitigation over user convenience. We simulated an **aggressive security posture** by lowering the threat threshold from `0.50` to `0.30`:

![Confusion Matrix Simulation](confusion_matrix_simulation.png)

### Simulation Metrics:
* **Standard Posture (Threshold = 0.50):**
  * Attacks Caught (True Positives): 16,168
  * Attacks Missed (False Negatives): 41
  * Legitimate Disrupted (False Positives): 80 (2.11%)
* **Aggressive Posture (Threshold = 0.30):**
  * Attacks Caught (True Positives): 16,181 (**Recall: 99.83%**)
  * Attacks Missed (False Negatives): **28** (31.7% reduction in missed threats)
  * Legitimate Disrupted (False Positives): **103 (2.72%)**

### Ethical Dilemma & SOC Trade-Offs:
1. **Self-Inflicted Denial of Service:** Lowering the threshold successfully catches more stealth attacks, but falsely blocks 103 legitimate user sessions (2.72% false alarm rate), causing business interruption.
2. **SOC Alert Fatigue:** Flooding analysts with false alarms causes operational burnout and risks operators muting alerts.
3. **Recommended Defense:** Implement a **tiered mitigation pipeline**:
   * $\ge 80\%$ confidence $\to$ Automated drop / quarantine.
   * $30\% - 79\%$ confidence $\to$ Step-up MFA, session rate-limiting, and analyst triage without terminating user access.

---

## 📈 Exploratory Data Analysis (EDA)

![EDA Plots](eda_plots.png)

- **Class Distribution:** The telemetry exhibits realistic real-world attack flooding (~81% attack traffic vs. ~19% normal baseline traffic in the evaluated flow sample).
- **Network Feature Profiling:** Network duration and flow characteristics exhibit distinct log-normal density profiles between benign connections and automated attack vectors.

---

## 🚀 Setup & Installation Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/DhruvArora664/ai-cybersecurity-unit1-mini-project-dhruv.git
cd ai-cybersecurity-unit1-mini-project-dhruv
```

### 2. Set Up Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Execute the Pipeline
You can run either the standalone script or view/run the executed Jupyter Notebook:

* **Run Python Script:**
  ```bash
  python src/main.py
  ```
* **Launch Jupyter Notebook:**
  ```bash
  jupyter notebook src/main.ipynb
  ```
*(Note: `src/main.ipynb` is pre-executed with all cell outputs, tables, and graphs visible directly on GitHub).*
