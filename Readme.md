# 🕵️‍♂️ Bitcoin Fraud Detection using Graph Neural Networks

This repository contains the machine learning pipeline for detecting illicit/fraudulent transactions in the **Elliptic Bitcoin Dataset**. The project leverages Graph Neural Networks (GNNs) and baseline models to handle highly imbalanced graph data.

## 📌 Project Overview
The Elliptic dataset consists of Bitcoin transactions with complex topological structures. Our goal is to classify transactions as either `licit` or `illicit`, handling a severe class imbalance and utilizing a strict temporal split for realistic evaluation.

🚀 Installation & Setup
We use Python 3.12.10 and PyTorch with CUDA 11.8 support for optimal GPU compatibility.
1. Prerequisites:
Ensure you have Python 3.12.10 installed on your system. You can verify this by running:
code
Bash
python --version
2. Clone the repository:
code
Bash
git clone https://github.com/your-username/bitcoin-fraud-detection.git
cd bitcoin-fraud-detection
3. Create and activate a Virtual Environment:
code
Bash
# Create the virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate