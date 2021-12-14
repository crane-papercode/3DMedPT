# 3DMedPT

**\[our code for IntrA and Modelnet40 classification is released]**

3D Medical Point Transformer: Introducing Convolution to Attention Networks for Medical Point Cloud
Analysis [[arxiv]](https://arxiv.org/pdf/2112.04863.pdf)

Author: Jianhui Yu, Chaoyi Zhang, Heng Wang, Dingxin Zhang, Yang Song, Tiange Xiang, Dongnan Liu, Weidong Cai

## Model Architecture

![model architecture](./images/model_details.jpg)

## Requirements

* Python >=3.6
* Pytorch >= 1.4
* Packages: tqdm, sklearn, visualdl, einops, natsort
* To build the CUDA kernel for FPS:
    ```angular2html
    pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
    ```

## Performance

* State-of-the-art accuracy on IntrA classification (F1 score): <b>0.936</b>
* State-of-the-art accuracy on IntrA segmentation (IoU): <b>94.82%</b> on healthy vessel and <b>82.39%</b> on aneurysm
* ModelNet40 classification: <b>93.4%</b>

## Citation

If you find our data or project useful in your research, please cite:

```
@article{yu20213d,
title={3D Medical Point Transformer: Introducing Convolution to Attention Networks for Medical Point Cloud Analysis},
author={Yu, Jianhui and Zhang, Chaoyi and Wang, Heng and Zhang, Dingxin and Song, Yang and Xiang, Tiange and Liu, Dongnan and Cai, Weidong},
journal={arXiv preprint arXiv:2112.04863},
year={2021}
}
```

### Acknowledgement

Our code is based on:

* [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)
* [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
* [dgcnn.pytorch](https://github.com/AnTao97/dgcnn.pytorch)