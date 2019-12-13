---
layout: default
title: Paper-Bookmark
has_children: true
nav_order: 2
permalink: docs/Paper-Bookmark
---

# Paper Notebook
{: .no_toc }

A collection of arbitrary kinds of computer vision papers, organized by [David Fan](https://github.com/davidhalladay).

Papers are ordered in arXiv submitting time (if applicable).

Feel free to send a PR or an issue.

{: .fs-6 .fw-300 }









## object detection & Segmentation

#### Classical

| V    | Model                                                        | Paper                                                        | Note                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
|      | [R-FCN](https://arxiv.org/pdf/1605.06409.pdf)                | [NIPS2016] R-FCN: Object Detection via Region-based Fully Convolutional Networks | ![12](./images/12.png)                                       |
|      | [R-FCN++](https://pdfs.semanticscholar.org/f4a2/732d4051b9c4b5d1f057aaa7935be390f51e.pdf) | [AAAI2018] R-FCN++: Towards Accurate Region-Based Fully Convolutional Networks for Object Detection | ![13](./images/13.png)                                       |
| V    | [Faster RCNN](https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf) | [NIPS2017] Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks | ![10](./images/10.png)                                       |
| V    | [Mask RCNN](https://arxiv.org/pdf/1703.06870.pdf)            | [ICCV2017] Mask R-CNN                                        | The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps<br />![09](./images/09.png) |


#### Relation Network

| V    | Model                                                        | Paper                                                        | Note                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
|      | [Paper](https://arxiv.org/pdf/1711.11575.pdf?fbclid=IwAR2SKIuG2_Izg7BNl6vhXBlAhkwEVxC2yt0ToP2R2CVg6IFKRqmqa-xd3C4) | [CVPR2018]Relation Networks for Object Detection             | This work proposes an object relation module. It processes a set of objects simultaneously through interaction between their appearance feature and geometry, thus allowing modeling of their relations |
| V    | [Reasoning-RCNN](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Reasoning-RCNN_Unifying_Adaptive_Global_Reasoning_Into_Large-Scale_Object_Detection_CVPR_2019_paper.pdf) | [CVPR2019]Reasoning-RCNN: Unifying Adaptive Global Reasoning into Large-scale Object Detection | ![03](./images/03.png)                                       |
|      |                                                              |                                                              |                                                              |

#### Tiny things
| V    | Model                                           | Paper                                                        | Note                                                         |
| ---- | ----------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
|      | [FPN](https://arxiv.org/pdf/1612.03144.pdf)     | [CVPR2017]Feature Pyramid Networks for Object Detection      |                                                              |
|      | [MDSSD](https://arxiv.org/pdf/1805.07009v2.pdf) | [2018]MDSSD: Multi-scale Deconvolutional Single Shot Detector for Small Objects | ![04](./images/04.png)                                       |
|      | [Paper](https://arxiv.org/pdf/1902.07296.pdf)   | [2019]Augmentation for small object detection                | We conjecture this is due to two factors; (1) only a few images are containing small objects, and (2) small objects do not appear enough even within each image containing them. We thus propose to oversample those images with small objects and augment each of those images by copy-pasting small objects many times |
|      | [Paper](https://arxiv.org/pdf/1901.01892.pdf)   | [ICCV2019] Scale-Aware Trident Networks for Object Detection | ![01](./images/01.png)                                       |
|      | [Paper](https://arxiv.org/pdf/1902.06042.pdf)   | [2019]R^2 -CNN: Fast Tiny Object Detection in Large-scale Remote Sensing Images |                                                              |


#### RPN
| V    | Model                                                        | Paper                                                        | Note                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
|      | [Faster-RCNN](https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks) | [NIPS2015]Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks |                                                              |
|      | [Anchor-free RPN](https://arxiv.org/abs/1804.09003)          | [2018] An Anchor-Free Region Proposal Network for Faster R-CNN based Text Detection Approaches | In order to better enclose scene text instances of various shapes, it requires to design anchors of various scales, aspect ratios and even orientations manually, which makes anchorbased methods sophisticated and inefficient<br />Compared with a vanilla RPN and FPN-RPN, AF-RPN can get rid of complicated anchor design and achieve higher recall rate on large-scale COCO-Text dataset. |
|      | [CRAFT](https://arxiv.org/pdf/1604.03239.pdf)                | [CVPR2016] CRAFT Objects from Images                         | ![08](./images/08.png)                                       |
|      | [Siamese RPN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf) | [CVPR2018] High Performance Visual Tracking with Siamese Region Proposal Network | ![11](./images/11.png)                                       |
|      | [Siamese cRPN](https://zpascal.net/cvpr2019/Fan_Siamese_Cascaded_Region_Proposal_Networks_for_Real-Time_Visual_Tracking_CVPR_2019_paper.pdf) | [CVPR2019] Siamese Cascaded Region Proposal Networks for **Real-Time Visual Tracking** | previously proposed one-stage Siamese-RPN trackers degenerate in presence of similar distractors and large scale variation. Addressing these issues, we propose a multi-stage tracking framework, Siamese Cascaded RPN (C-RPN), which consists of a sequence of RPNs cascaded from deep high-level to shallow low-level layers in a Siamese network. |
|      | [Enhanced-PRN](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0203897&type=printable) | [PLOS2018] An Enhanced Region Proposal Network for object detection using deep learning method |                                                              |
|      | [GA-Nets](https://arxiv.org/pdf/1901.03278.pdf)              | [CVPR2019] Region Proposal by Guided Anchoring               |                                                              |



## Affinity Matrix

| V    | Model                                           | Paper                          | Note                                                         |
| ---- | ----------------------------------------------- | ------------------------------ | ------------------------------------------------------------ |
|      | [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Spatial-Aware_Graph_Relation_Network_for_Large-Scale_Object_Detection_CVPR_2019_paper.pdf) | [CVPR2019] Spatial-aware Graph Relation Network for Large-scale Object Detection | ![14](images/14.png) | |

## Relationship Learning

| V    | Model                                                        | Paper                                                        | Note                 |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------- |
|      | [Paper](http://vipl.ict.ac.cn/uploadfile/upload/2018122017585071.pdf) | [AAAI2019] Deep Structured Learning for Visual Relationship Detection | ![14](images/14.png) |




## Compressed sensing

| V    | Model                                           | Paper                                             | Note                                                         |
| ---- | ----------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------ |
|      | [Paper](https://arxiv.org/pdf/1703.03208.pdf)   | [2017] Compressed Sensing using Generative Models |                                                              |
|      | [Deep-CS](https://arxiv.org/pdf/1905.06723.pdf) | [2019] Deep Compressed Sensing                    | This paper proposes a deep background subtraction method based on conditional Generative Adversarial Network (cGAN) |
|      |                                                 |                                                   |                                                              |



## other

| V    | Model                                                        | Paper                                                        | Note                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
|      | [BSC-GAN](https://www.researchgate.net/publication/327630853_BSCGAN_Deep_Background_Subtraction_with_Conditional_Generative_Adversarial_Networks) | [ICIP2018] Deep Background Subtraction with Conditional Generative Adversarial Networks | This paper proposes a deep background subtraction method based on conditional Generative Adversarial Network (cGAN) |
|      | [Paper](http://mi.eng.cam.ac.uk/~cipolla/publications/inproceedings/2018-CVPR-multi-task-learning.pdf?fbclid=IwAR2alX5p6v2xVq2GFSPri2mAov7UFK8S8egOWWhsOd29k1JRockLjBazGxI) | [CVPR2018] Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics |                                                              |
|      | [Paper](https://kgtutorial.github.io)                        | [CVPR2018]Relation Networks for Object Detection             | ![01](./images/02.png)                                       |
|      | [Paper](https://zpascal.net/cvpr2019/Chen_Multi-Label_Image_Recognition_With_Graph_Convolutional_Networks_CVPR_2019_paper.pdf) | [CVPR2019]Multi-Label Image Recognition with Graph Convolutional Networks∗ |                                                              |
|      | [Paper](https://arxiv.org/pdf/1909.09953.pdf)                | [2019] Learning Visual Relation Priors for Image-Text Matching and Image Captioning with Neural Scene Graph Generators |                                                              |
|      | [Paper](https://arxiv.org/pdf/1909.05370.pdf)                | [2019] An Auxiliary Classifier Generative Adversarial Framework for Relation Extraction |                                                              |
|      | [Paper]([chrome-extension://bjfhmglciegochdpefhhlphglcehbmek/content/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F1802.10171.pdf](chrome-extension://bjfhmglciegochdpefhhlphglcehbmek/content/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F1802.10171.pdf)) | Tell Me Where to Look: Guided Attention Inference Network    |                                                              |

## Meeting

| V    | Model                                                        | Paper                                                        | Note |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
|      | [Paper](https://openreview.net/pdf?id=HJlqN4HlIB)            | [NIPS2019] Deep Learning without Weight Transport            |      |
|      | [Paper](http://papers.nips.cc/paper/8885-weakly-supervised-instance-segmentation-using-the-bounding-box-tightness-prior.pdf) | [NIPS2019] Weakly Supervised Instance Segmentation using the Bounding Box Tightness Prior |      |
