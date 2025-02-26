# Simple-TMF-Matting

Inofficial simplified version of [TMF-Matting](https://github.com/Serge-weihao/TMF-Matting) with minimal dependencies.

Given an image and a trimap, compute its alpha matte.

| **Input image** | **Input trimap** | **Output alpha matte** |
|--------------------------|-----------------|------------------|
| ![image](https://raw.githubusercontent.com/frcs/alternative-matting-laplacian/master/GT04.png) | ![trimap](https://raw.githubusercontent.com/frcs/alternative-matting-laplacian/master/trimap-GT04.png) | ![alpha](https://github.com/user-attachments/assets/f7b70881-0c60-4ec5-a7d6-d820ebc4dddf) |

The test image is from https://alphamatting.com/datasets.php.

# Usage

1. Download the pretrained `comp1k.pth` model [from the original authors' repository](https://github.com/Serge-weihao/TMF-Matting?tab=readme-ov-file#results-and-models) and place it in this directory.
2. Run [`test_single_image.py`](https://github.com/99991/Simple-TMF-Matting/blob/main/test_single_image.py), which will download test images and compute the alpha matte.

# Citing

If you find TMFNet useful in your research, please consider citing [the original authors](https://github.com/Serge-weihao/TMF-Matting?tab=readme-ov-file#citing):

```BibTex
@article{jiang2023trimap,
  title={Trimap-guided feature mining and fusion network for natural image matting},
  author={Jiang, Weihao and Yu, Dongdong and Xie, Zhaozhi and Li, Yaoyi and Yuan, Zehuan and Lu, Hongtao},
  journal={Computer Vision and Image Understanding},
  volume={230},
  pages={103645},
  year={2023},
  publisher={Elsevier}
}
```
