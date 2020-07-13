# GCN-Denoiser: Mesh Denoising with Graph Convolutional Networks
Qt and Pytorch implementation for GCN-Denoiser

### Denoised Results:

![](/imgs/result.png)

### Interface:

<img src="/imgs/new_interface.png" alt="interface|60%" style="zoom:50%;" />

## Code:

### Prerequisites:

- Hardware: Personal computer with NVIDIA GPU.
- Environments: CUDA10.0, Windows system (network training part can also be used on Linux).

### Third Party Library:

- [Pytroch C++ 1.2.0](https://pytorch.org/) , [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page), [Flann](https://github.com/mariusmuja/flann) and [OpenMesh](https://www.graphics.rwth-aachen.de/software/openmesh/) at runtime.
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

`bunny_0_2/*.mat` are sampled patches from the noisy *bunny* model with 0.2 level of Gaussian noise.

### Denoising Interface:

Executable demo, the corresponding code, and some sampled meshes are supplied. *New simplified version has been updated*

- For .exe, windows platform is required and the CUDA PATH must be set in the system environment. Some important `.dll` have been supplied ( Unzip dlls.zip firstly).

- For code, Visual Studio 2017 and Qt 5.12 are required.

### Pre-trained models:

One version of GCN pre-trained model for synthetic models is supplied.

### Acknowledgements

Part of this implementations is based on [DGCNN](https://github.com/WangYueFt/dgcnn) and [GNF](https://github.com/bldeng/GuidedDenoising).

### Keep Updating...