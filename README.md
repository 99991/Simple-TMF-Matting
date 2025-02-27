# Simple-TMF-Matting

Inofficial simplified version of [TMF-Matting](https://github.com/Serge-weihao/TMF-Matting) with minimal dependencies.

Given an image and a trimap, compute its alpha matte.

| **Input image** | **Input trimap** | **Output alpha matte** |
|--------------------------|-----------------|------------------|
| ![image](https://raw.githubusercontent.com/frcs/alternative-matting-laplacian/master/GT04.png) | ![trimap](https://raw.githubusercontent.com/frcs/alternative-matting-laplacian/master/trimap-GT04.png) | ![alpha](https://github.com/user-attachments/assets/f7b70881-0c60-4ec5-a7d6-d820ebc4dddf) |

The test image is from https://alphamatting.com/datasets.php.

# Usage

1. [Install PyTorch](https://pytorch.org/get-started/locally/)
2. Install Pillow: `pip install pillow`
3. Download the pretrained `comp1k.pth` model [from the original authors' repository](https://drive.google.com/file/d/1zTEYBXaAlEU-nt703W9OFRNchfabEOxs/view) and place it in this directory.
4. Run [`test_single_image.py`](https://github.com/99991/Simple-TMF-Matting/blob/main/test_single_image.py), which will download test images and compute the alpha matte.

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

# Testing on Composition-1K Test Set

1. Download the pretrained model as above.
2. Ask [Brain Price](https://arxiv.org/pdf/1703.03872) to send you `Adobe_Deep_Matting_Dataset.zip` and place it in this directory. Do not unzip.
3. Download and extract the images of the [Pascal VOC2012 dataset](host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit) to the directory `PascalVOC2012`. You can also link them with `ln -s YOUR_PASCAL_DIR/VOCdevkit/VOC2012/JPEGImages/ PascalVOC2012` if you already have them somewhere else.
4. Run [`test_composition_1k_dataset.py`](https://github.com/99991/Simple-TMF-Matting/blob/main/test_composition_1k_dataset.py)

# [Results](https://github.com/99991/Simple-TMF-Matting/blob/main/test_composition_1k_dataset.py)

| MSE × 1000 | SAD / 1000 |
| ----- | ------ |
| 4.547 | 22.410 |

MSE is slightly worse and SAD is slightly better than original, but minor details such as background interpolation method result in a large difference, so this is probably acceptable.
