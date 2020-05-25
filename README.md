# GCN-Denoiser: Mesh Denoising with Graph Convolutional Networks
Qt and Pytorch implementation for GCN-Denoiser

### Denoised Results:

![](/imgs/result.png)

### Interface:

![](/imgs/interface.png)

## Code:

### Prerequisites:

- Hardware: Personal computer with NVIDIA GPU.
- Environments: CUDA10.0, Windows system (network training part can also be uesd on Linux).

### Third Party Library:

- [Pytroch C++ 1.2.0](https://pytorch.org/) , [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) and [OpenMesh](https://www.graphics.rwth-aachen.de/software/openmesh/) at runtime.
- Pytorch 1.2.0, numpy, Scipy 1.4.1 and tensorbordx 1.13 (\>python3.5) in training stage.

### Network part:

The training code and part of validation data are supplied. Network test can be run by:

```
cd DenoisingGCN/testSamples
unzip bunny_0_2.zip
cd ../
python datautils.py
python test.py
```

bunny_0_2/*.mat are sampled patches from the noisy *bunny* model with 0.2 level of Gaussion noise.

### Denoising Interface:

Executable demo, the corresponding code, and some sampled mesh are supplied.

- For code, Visual Studio 2017 and Qt 5.12 are required.

### Pre-trained models:

One version of GCN pre-trained model for synthetic models is supplied.

### Acknowledgements

Part of this implementations is based on [DGCNN](https://github.com/WangYueFt/dgcnn) and [GNF](https://github.com/bldeng/GuidedDenoising).

### Keep Updating...