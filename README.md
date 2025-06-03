# 🧬 Quantum Machine Learning for Predicting Binding Free Energies in Structure-Based Virtual Screening


---

## 📌 Project Information

- **Title**: Quantum Machine Learning for Predicting Binding Free Energies in SBVS  
- **Author**: Pei-Kun Yang  
- **Contact**: [peikun@isu.edu.tw](mailto:peikun@isu.edu.tw)

---

## ⚙️ Installation

We recommend using a virtual environment. To install the required packages:

```bash
pip install numpy torch pennylane

## 📁 Project Structure

The repository contains the following directories:

- **1_database/**  
  Prepares the datasets for training and testing, including molecular structures and labels.

- **2_train/**  
  Contains the training scripts for optimizing quantum circuit parameters using PennyLane and PyTorch.

- **3_check_par/**  
  Loads the datasets from `1_database` and the trained parameters from `2_train`.  
  Uses PennyLane in *probability mode* to compute RMSD and Pearson correlation between predicted and true values.

- **4_shot/**  
  Similar to `3_check_par`, but uses *finite shot sampling* mode in PennyLane for evaluating RMSD and Pearson correlation.

- **5_noise/**  
  Adds quantum noise models to the circuit to simulate more realistic quantum hardware behavior.

- **6_ana/**  
  Analyzes and visualizes the results (e.g., error trends, performance metrics).

