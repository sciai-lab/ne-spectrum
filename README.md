![PyPI - Version](https://img.shields.io/pypi/v/ne-spectrum)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Neighbor embedding spectrum

This repository implements the computation of neighbor embedding spectra as described in 
*Visualizing single-cell data with the neighbor embedding spectrum* ([bioarxiv](https://www.biorxiv.org/content/10.1101/2024.04.26.590867v1)).

It can use [`openTSNE`](https://pypi.org/project/openTSNE/) or [`cne`](https://pypi.org/project/contrastive-ne/) as backends.


# Installation
```bash
pip install ne-spectrum
```

If you want to use the GPU support for the `cne` backend, please make sure that you have pytorch installed with CUDA support before installing `ne-spectrum`.
Similarly, if you want to save animations as `.mp4` rather than `.gif` files, you need to install [ffmpeg](https://ffmpeg.org/).
# Usage

Load MNIST as example data
```python
from ne_spectrum import TSNESpectrum, CNESpectrum
import torchvision
from sklearn.decomposition import PCA
import os
import numpy as np

fig_path = "./"

# load MNIST as example dataset
mnist_train = torchvision.datasets.MNIST(train=True,
                                         download=True,
                                         transform=None,
                                         root=fig_path
                                         )
x_train, y_train = mnist_train.data.float().numpy(), mnist_train.targets

mnist_test = torchvision.datasets.MNIST(train=False,
                                        download=True,
                                        transform=None,
                                        root=fig_path
                                        )
x_test, y_test = mnist_test.data.float().numpy(), mnist_test.targets

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

x = np.concatenate([x_train, x_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

# transform data with PCA to save some time when computing the kNN graphs
x_pca = PCA(n_components=50).fit_transform(x)
```

Compute the neighbor embedding spectrum with the `openTSNE` backend
```python
# compute spectrum with openTSNE backend
tsnespectrum = TSNESpectrum()
tsnespectrum.fit(x_pca)

# save individual slides, all embeddings, and a gif animation
tsnespectrum.save_slides(save_path=os.path.join(fig_path, "mnist_tsne"),
                         cmap="tab10",
                         color=y)
tsnespectrum.save_embeddings(os.path.join(fig_path, "mnist_tsne", "embeddings.npy"))
tsnespectrum.save_video(save_path=os.path.join(fig_path, "mnist_tsne"),
                        cmap="tab10",
                        color=y)
```
<p align="center"><img  alt="Neighbor embedding spectrum on MNIST animated" src="/mnist_tsne_spectrum.gif" width="600"/>



Similarly, we can compute the neighbor embedding spectrum with the `cne` backend
```python
# compute spectrum with CNE backend
cnespectrum = CNESpectrum()
cnespectrum.fit(x_pca)

# save individual slides, all embeddings, and a gif animation
cnespectrum.save_slides(save_path=os.path.join(fig_path, "mnist_cne"),
                        cmap="tab10",
                        color=y)
cnespectrum.save_embeddings(os.path.join(fig_path, "mnist_cne", "embeddings.npy"))
cnespectrum.save_video(save_path=os.path.join(fig_path, "mnist_cne"),
                       cmap="tab10",
                       color=y)
```
