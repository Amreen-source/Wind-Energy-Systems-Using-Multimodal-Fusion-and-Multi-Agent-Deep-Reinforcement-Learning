# Intelligent Predictive Maintenance for Wind Turbines

### A Multimodal & Multi-Agent Reinforcement Learning Framework

This repository contains an end-to-end system for predictive maintenance in wind farms.
The goal is to use information from several sources—sensor readings, drone imagery, technician notes, and weather—to estimate the condition of each turbine and decide when maintenance should happen.

The framework combines deep learning for feature extraction and multi-agent reinforcement learning (MARL) for coordinated decision-making across a farm.

---

## 🔍 What This Project Does

* **Collects & processes four different data types**

  * SCADA time-series
  * Blade inspection images
  * Written maintenance reports
  * Local weather conditions

* **Transforms each data type using a specialized encoder**

  * TCN → time-series patterns
  * Vision Transformer → blade defects
  * BERT → technician notes
  * MLP → environmental context

* **Fuses the four feature sets with a cross-attention module**, giving the system a rich and unified view of turbine state.

* **Uses multiple reinforcement-learning agents**, one per turbine:

  * Agents communicate with neighbors
  * A QMIX network coordinates their decisions
  * Agents choose actions like "inspect", "minor repair", "major repair", or "wait"

* **Optimizes long-term maintenance strategy**, reducing:

  * unplanned failures
  * maintenance frequency
  * energy loss
  * overall operating cost

---

## 🧱 System Overview (Plain English)

1. Each turbine has its own “agent”.
2. The agent observes its turbine’s fused multimodal state.
3. Agents exchange small messages with nearby turbines.
4. Each agent outputs a recommended maintenance action.
5. A mixing network checks that all actions work well together.
6. The environment simulates cost, downtime, and health impact.
7. The system learns the best long-term maintenance scheduling policy.

---

## 📂 Project Structure

```
├── data/                # Sample synthetic dataset & loaders
├── encoders/            # TCN, ViT, BERT, MLP implementations
├── fusion/              # Cross-attention multimodal fusion layer
├── marl/                # Q-networks, communication, QMIX mixing
├── environment/         # Maintenance simulation and reward logic
├── training/            # Scripts for training MARL agents
├── utils/               # Plots, metrics, logging helpers
└── README.md
```

---

## 🚀 Getting Started

### Install Dependencies

```
pip install -r requirements.txt
```

### Train the MARL Model

```
python training/train_marl.py --config configs/default.yaml
```

### Run Evaluation

```
python evaluate.py --checkpoint results/best_model.pth
```

---

## 📊 Expected Outcomes

Although this repo uses synthetic or openly available datasets,
the system demonstrates:

* reduced maintenance cost
* lower failure frequency
* better turbine availability
* more consistent scheduling
* improved decision quality using multimodal data

Performance will vary depending on data and environment configuration.

---

## ✨ Why This Framework Is Useful

Most wind-turbine health-monitoring tools use only one data type (usually SCADA).
This project shows how multiple information sources can be combined to:

* detect early degradation
* reduce unnecessary maintenance
* schedule repairs when weather and load conditions are favorable
* coordinate decisions across the entire wind farm

It serves as a blueprint for real-world predictive-maintenance systems.

---

## 📜 License

MIT License — free to modify and use.

---

## 🙌 Acknowledgement

This implementation is inspired by modern multimodal deep-learning methods and multi-agent reinforcement learning research.
Certain parts of the conceptual design are based on academic literature, but the code and README are written independently.
