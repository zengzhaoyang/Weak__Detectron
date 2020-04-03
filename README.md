# WSOD^2: Learning Bottom-up and Top-down Objectness Distillation forWeakly-supervised Object Detection

By Zhaoyang Zeng, Bei Liu, Jianlong Fu, Hongyang Chao and Lei Zhang

### Introduction

**WSOD^2** is a framework for weakly supervised object detection.
  - It achieves state-of-the-art performance on weakly supervised object detection (Pascal VOC 2007 and 2012)
  - Our code is written by Pytorch, based on [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch)
  - We also implement [WSDDN]() and [OICR]() for easier comparasion.

The paper has been accepted by ICCV 2019. For more details, please refer to our [paper](https://arxiv.org/pdf/1909.04972)

### Architecture

### License

WSOD^2 is released under the MIT License (refer to the LICENSE file for details).

### Citing WSOD^2

If you find WSOD^2 useful in your research, please consider citing:

@inproceedings{zeng2019wsod2,
  title={WSOD2: Learning Bottom-Up and Top-Down Objectness Distillation for Weakly-Supervised Object Detection},
  author={Zeng, Zhaoyang and Liu, Bei and Fu, Jianlong and Chao, Hongyang and Zhang, Lei},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={8292--8300},
  year={2019}
}

### Installation

1. Clone the WSOD^2 repository
```shell
git clone https://github.com/zengzhaoyang/Weak__Detectron
```

2. Install required packages
```shell
pip install -i requirements.txt
```

3. Build the C-based modules
```shell
cd lib && bash make.sh
```

### Data

1. Download the training, validation, test data and VOCdevkit

```shell
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
```

Then extract them in "data/" folder

2. The folder structure should be
```shell
data/VOCdevkit/
data/VOCdevkit/VOC2007
data/VOCdevkit/VOC2007/Annotations
data/VOCdevkit/VOC2007/JPEGImages
data/VOCdevkit/VOC2007/ImageSets
# ... and several other directories ...
```

3. Download the coco-style voc annotations, pre-computed selective search proposals and VGG16 Imagenet pre-trained model.
```shell
wget ***
```

### Usage

1. **Training** 
```shell
python tools/train.py --cfg configs/wsod2.yaml
```

2. **Testing**
```shell
python tools/test.py --cfg configs/wsod2.yaml --load_ckpt [path to checkpoint]
```


