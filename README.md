# GCN-Denoiser: Mesh Denoising with Graph Convolutional Networks
Qt and Pytorch implementation for our paper "[GCN-Denoiser: Mesh Denoising with Graph Convolutional Networks](http://www.youyizheng.net/docs/gcn-denoiser.pdf)" (ACM Transactions on Graphics 2022)

We propose GCN-Denoiser, a novel feature-preserving mesh denoising method based on graph convolutional networks (GCNs). Unlike previous learning-based mesh denoising methods that exploit hand-crafted or voxel-based representations for feature learning, our method explores the structure of a triangular mesh itself and introduces a graph representation followed by graph convolution operations in the dual space of triangles. We also create a new dataset called PrintData containing 20 real scans with their corresponding ground truths for the research community.

### Denoised Results:

<img src="/imgs/result.png" style="zoom:30%;" />

### Interface:

<img src="/imgs/interface.png" style="zoom:60%;" />

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

Executable demo, the corresponding code, and some sampled meshes are supplied.

- For .exe, windows platform is required and the CUDA PATH must be set in the system environment. Some `.dll` are required (CUDA&LibTorch: c10.dll, c10_cuda.dll, caffe2_nvrtc.dll, nvToolsExt61_1.dll, torch.dll; Qt: Qt5Core.dll, Qt5Gui.dll, Qt5OpenGL.dll, Qt5Widgets.dll).

- For code, Visual Studio 2017 and Qt 5.12 are required.

### Pre-trained models:

One version of GCN pre-trained model for synthetic models is supplied.

## Dataset:

<img src="/imgs/printeddataset.png" style="zoom:30%;" />

See the zipped file "PrintedDataset.zip". 

### Citation

If you find this useful for your research, please cite the following paper.

```
@article{shen2022gcndenoiser,
  title={GCN-Denoiser: Mesh Denoising with Graph Convolutional Networks},
  author={Shen, yuefan and Fu, Hongbo and Du, Zhongshuo and Chen, Xiang and Burnaev, Evgeny and Zorin, Denis and Zhou, Kun and Zheng, Youyi},
  journal={ACM Trans. Graph.},
  volume={41},
  number={1},
  issn={0730-0301},
  numpages={14},
  year={2022}
}
```

Waiting for updating...

### Acknowledgements

Part of this implementations is based on [DGCNN](https://github.com/WangYueFt/dgcnn) and [GNF](https://github.com/bldeng/GuidedDenoising).
