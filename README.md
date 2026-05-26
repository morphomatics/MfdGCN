# Manifold GCN

Manifold GCN is a general-purpose neural network for deep learning on graphs with manifold-valued node features. 
It is well-suited for tasks like segmentation, classification, feature extraction, etc. It was presented in the paper:

> Martin Hanik, Gabriele Steidl, Christoph von Tycowicz:  
> **[Manifold GCN: Diffusion-based Convolutional Neural Network for Manifold-valued Graphs.]()**  
> International Journal of Computer Vision, Volume 134, 2026. </br>
> [![DOI](https://img.shields.io/badge/DOI-xxx-yellow)](http://dx.doi.org/10.1137/21M1410373) [![Preprint](https://img.shields.io/badge/arXiv-2402.12901-red)](https://arxiv.org/abs/2401.14381)

The network layers can be found in the [Morphomatics](https://morphomatics.github.io/) library. This repository contains 
the code for the experiments in the paper.

# Citation

If you use this implementation in your academic projects, we politely ask you to acknowledge it in your manuscript by 
citing the paper:

```bibtex
@article{Hanik2026mfdgcn,
  author = {Martin Hanik and Gabriele Steidl and Christoph von Tycowicz},
  title = {Manifold GCN: Diffusion-based Convolutional Neural Network for Manifold-valued Graph},
  journal = {International Journal of Computer Vision},
  volume = {134},
  pages = {x--y},
  year = {2026},
  publisher = {Springer}
}
```

# Experiments

To run experiments unsing Manifold GCN, you need to (pip-)install the following packages:
- jax[cuda],
- morphomatics==4.1.3,
- flax,
- optax,
- jraph,
- pyvista,
- networkx,
- sckit-learn.

[Diffusion Net](https://github.com/nmwsharp/diffusion-net) and [Mesh CNN](https://github.com/ranahanocka/MeshCNN/) are 
not included in these requirements. They must be installed separately and require additional dependencies. We refer to the 
respective repositories for installation instructions. Depending on your setup, there might be problems to install 
Jax and Pytorch for GPU at the same time. In that case, we recommend to use different virtual environments when training
Jax- and Pytorch-based models.

This code was tested with Python 3.11 and CUDA 12.1.

## Synthetic Graph Classification

### Manifold GCN

To run the experiment on synthetic graphs, navigate to the `experiments/synthetic_graphs` folder and run the following 
command:

```
python synthetic_data_classification.py --graphs_per_class 30 --feature_space hyperbolic --initial_embedding one_hot
```

This will generate 30 synthetic graphs of each class and run the experiment on the hyperbolic manifold with one-hot 
encoding. To embedd in the manifold of symmetric positive definite matrices, use `spd` instead of `hyperbolic`. For the 
degree embedding, use `degree` instead of `one_hot`. This works only in hyperbolic space.

### Comparison methods

H2H-GCN can be run (on 90 graphs) with the command:

```
python synthetic_data_classification_h2h.py --graphs_per_class 30
```

For HGNN, we refer to the [repository](https://github.com/facebookresearch/hgnn) of the original paper.

## Alzheimer Classification from Hippocampi

### Manifold GCN

To run our Manifold GCN for Alzheimer classification, navigate to the `experiments/adni` folder. Use the following 
commands to run the experiment with normals and differential-coordinate features, respectively:

```
python adni_classification_s2.py
```

```
python adni_classification_shape_space.py
```

### Comparison methods

To use Diffusion Net, first clone the [repository](https://github.com/nmwsharp/diffusion-net) to the location of your
choice and add the path to the source (usually `diffusion-net/src`) folder to the `PYTHONPATH` environment variable. 
When the repository was cloned into the `adni` folder, the following command can be used to run Diffusion Net with 
xyz-coordinate features:

```
PYTHONPATH=./diffusion-net/src python adni_classification_diffusion_net.py --features xyz
```

Using `hks`, `normal`, or `dcm`, the network receives heat kernel signatures, normals, or differential-coordinate 
features, respectively.

