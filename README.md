# A-Bayesian-Federated-Learning-Framework-with-Online-Laplace-Approximation

Brief: this repository contains PyTorch implementations for several federated learning baselines and variants (FedAvg, FedProx, CURV, and the authors' Bayesian/Online-Laplace method). The code trains small CNNs on standard vision datasets (CIFAR-10, CIFAR-100, MNIST) using a Dirichlet partitioning to simulate non-iid clients.

---

## 🔧 Requirements & recommended setup

- OS: Linux (tested)
- Python: 3.8+ (create a virtual environment)
- GPU recommended (CUDA-supported PyTorch)

Install dependencies (recommended):

1) Install PyTorch + torchvision first — choose the command for your CUDA version at https://pytorch.org/get-started/locally/.

Example (CPU / no-GPU):

    pip install torch torchvision

2) Install the other Python dependencies:

    pip install -r requirements.txt

Files that require these packages: `numpy`, `Pillow` (PIL), `torchvision`, and `wandb` (optional).

> Note: `torch` is not pinned in `requirements.txt` because the correct wheel depends on your CUDA setup. Follow the PyTorch site.

---

## 🚀 Quick start — run an experiment

From the repository root:

- Run the authors' method (default: CIFAR-10):

    python cnn_lab/cnn_ours.py --data cifar10 --n_round 20 --n_client 20 --n_epoch 40 --lr 0.01 --alpha 0.01 --train_batch_size 32 --gpu 0

- Run FedAvg baseline:

    python cnn_lab/cnn_fedavg.py --data cifar10 --n_round 20 --n_client 20 --n_epoch 40 --lr 0.01 --alpha 0.01 --train_batch_size 32 --gpu 0

- Run FedProx / CURV / offline variants:

    python cnn_lab/cnn_prox.py [args]
    python cnn_lab/cnn_curv.py [args]
    python cnn_lab/cnn_offline.py [args]

- The supplied bash scripts in the repo (`cifar10_lab_e10.sh`, `cifar10_lab_e40.sh`, `cifar10_lab_e80.sh`, `cifar10_lab_e160.sh`) show example hyperparameter sweep loops.

---

## ⚙️ Important CLI flags (common)

- `--data` : dataset name (`cifar10`, `cifar100`, `mnist`)  (default `cifar10` in most scripts)
- `--n_round` : number of federated rounds
- `--n_client` : number of simulated clients
- `--n_epoch` : local epochs per client
- `--lr` : learning rate
- `--alpha` : Dirichlet alpha (controls heterogeneity)
- `--train_batch_size` : client batch size
- `--gpu` : CUDA device id (set `--gpu='0'` to use GPU 0)
- `--csd_importance` : regularizer strength used by Bayesian / Laplace variants
- `--pruing_p` : pruning fraction (if used)

Run `python cnn_lab/<script>.py --help` for script-specific flags and defaults.

---

## 📂 Data / outputs

- Datasets are downloaded to `./data/` by `torchvision.datasets`.
- Logs: `./cnn_lab/hyperlog/` (each run writes a `.log`).
- Checkpoints: `./checkpoint/` (scripts create `../checkpoint/` relative to `cnn_lab/`).

---

## 🔍 Notes & known issues (important)

- model import caveat: `model.py` contains `from ShuffleNet import shufflenet` at the top. If you don't have a `ShuffleNet` module installed, importing `model.py` may raise ImportError. The repository does not call `shufflenet()` by default — you can safely ignore or remove the import if you won't use ShuffleNet.

- wandb: several scripts call `wandb.login(key="put your wandb key here")`. Replace that placeholder with your API key or set the environment variable `WANDB_API_KEY` and remove/skip the hard-coded login line. Example:

    export WANDB_API_KEY="<your-key>"

- Default `--gpu` value in scripts is `'2'`. Explicitly set `--gpu` or `CUDA_VISIBLE_DEVICES` if you have a different GPU layout.

---

## ✅ Example minimal run (GPU)

1) create venv and install:

    python -m venv venv
    source venv/bin/activate
    # install torch using instructions for your CUDA version
    pip install torch torchvision
    pip install -r requirements.txt

2) run one epoch with small config:

    python cnn_lab/cnn_ours.py --data cifar10 --n_round 5 --n_client 5 --n_epoch 1 --lr 0.01 --alpha 0.1 --train_batch_size 32 --gpu 0

---

## 🛠 Troubleshooting

- Dataset download fails: check network or manually download datasets into `./data/`.
- ImportError for `ShuffleNet`: either install that module or edit `model.py` to remove/guard the import.
- wandb errors: remove `wandb.login(...)` from files or set `WANDB_API_KEY`.

---

## 📚 Files of interest

- `cnn_lab/cnn_fedavg.py` — FedAvg baseline (example usage shown above)
- `cnn_lab/cnn_prox.py` — FedProx-style local objective
- `cnn_lab/cnn_curv.py` — CURV variant
- `cnn_lab/cnn_ours.py` — authors' Bayesian / Online-Laplace aggregation
- `cnn_lab/cnn_offline.py` — offline oracle-style aggregation
- `dirichlet_data.py` — data partitioning + dataset transforms (uses `autoaugment.py`)
- `model.py` — model definitions (BasicCNN, VGG9, MLP)
- `cnn_lab/autoaugment.py` — Cutout and AutoAugment policies used for CIFAR transforms
