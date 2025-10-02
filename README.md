# MLP Experiments

This repository contains code for running experiments with Multi-Layer Perceptrons (MLPs) on different synthetic datasets.  
The project uses **TensorFlow/Keras** for model training and **Hydra** for configuration management.

---

## Project structure

```
mlp_experiments/
│
├── configs/                 # Hydra configuration files
│   ├── config.yaml           # main config (defaults for model, training, plots, etc.)
│   └── data/                 # dataset-specific configs
│
├── src/                     # source code
│   ├── main.py               # entry point for running experiments
│   ├── data_loader.py        # dataset registry and preprocessing
│   ├── datasets.py           # definitions of synthetic datasets
│   ├── models.py             # MLP model builder
│   ├── experiment_logger.py  # JSON logging of runs
│   ├── run_plots.py          # functions to generate plots
│   └── utils.py              # callbacks, learning rate schedules, helpers
│
├── outputs/                  # automatically created by Hydra (ignored by git)
├── requirements.txt          # list of dependencies
└── README.md
```

---

## Installation

Clone the repository and install the required Python packages.

```bash
git clone https://github.com/YOURUSER/mlp_experiments.git
cd mlp_experiments

python -m venv .venv
.venv\Scripts\activate   # on Windows
# source .venv/bin/activate  # on Linux/Mac

pip install -r requirements.txt
```

---

## Usage

Run experiments with:

```bash
python src/main.py
```

You can override settings directly from the command line. For example:

- Change number of epochs:
  ```bash
  python src/main.py train.epochs=50
  ```

- Select another dataset:
  ```bash
  python src/main.py data=complex
  ```

- Run multiple experiments at once (sweep):
  ```bash
  python src/main.py -m train.epochs=20,50,100
  ```

Results, logs, and plots are saved in the `outputs/` directory, inside subfolders created automatically for each run.

---

## Outputs

Each run produces:

- `config.json` – configuration used for the run  
- `data_meta.json` – dataset information  
- `model.json` – model details and parameters  
- `epochs.jsonl` – metrics for each epoch  
- `weights_snapshots/` – weight matrices saved at intervals  
- `artifacts/` – plots such as accuracy, loss, and confidence curves  

---

## Requirements

See `requirements.txt` for dependencies.  
Main packages:
- TensorFlow
- Hydra
- scikit-learn
- matplotlib
- pandas
- numpy

---
