# Empowering Active Learning for 3D Molecular Graphs with Geometric Graph Isomorphism

Code for our NeurIPS 2024 paper.

[[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6462073c6bdf864ebfbbb11e80619f3e-Abstract-Conference.html) [[OpenReview]](https://openreview.net/forum?id=na7AgFyp1r)

## Overview

We propose an active learning framework for 3D molecular property prediction. The key idea is to jointly optimize uncertainty and diversity when selecting molecules for labeling, reducing the cost of expensive quantum mechanical calculations.

Three main components:

- **Bayesian Geometric GNN**: A SphereNet variant with Concrete Dropout for uncertainty estimation on 3D molecular graphs.
- **Geometric Graph Isomorphism**: A set of 3D graph isometries based on distributions of bond lengths, angles, and dihedral angles. Provably at least as expressive as the Geometric Weisfeiler-Lehman test. Moments of these distributions serve as molecular descriptors for diversity computation.
- **QP-based Selection**: Sample selection formulated as quadratic programming, balancing uncertainty and diversity.

## Setup

```bash
pip install -r requirements.txt
```

Requires PyTorch, PyTorch Geometric, and related libraries (torch-scatter, etc.). See [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for platform-specific instructions.

## Usage

```bash
# Example: run active learning with MC Dropout selection on QM9
python main.py --device 0 --selection_method mcdrop --cycle 0 --expt 1

# Available selection methods:
#   random, mcdrop, coreset, lloss, unc_div (ours)
```

The QM9 dataset will be automatically downloaded to `dataset/` on first run.

See `run.sh` for more examples.

## Project Structure

```
al/                  # Active learning methods (selection, loss prediction, etc.)
dig/                 # DIG library (3D graph models: SphereNet, SchNet, DimeNet++, etc.)
main.py               # Main entry point
```

## Citation

```bibtex
@inproceedings{subedi2024empowering,
  title={Empowering Active Learning for 3D Molecular Graphs with Geometric Graph Isomorphism},
  author={Subedi, Ronast and Wei, Lu and Gao, Wenhan and Chakraborty, Shayok and Liu, Yi},
  booktitle={Advances in Neural Information Processing Systems},
  volume={37},
  year={2024}
}
```

## License

GPLv3. See [LICENSE](LICENSE) for details.
