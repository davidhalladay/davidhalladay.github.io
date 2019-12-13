---
layout: default
title: Object-Detection
parent: Paper-Bookmark
nav_order: 5
---

# object detection & Segmentation

## Classical

| V    | Model                                                        | Paper                                                        | Note                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
|      | [R-FCN](https://arxiv.org/pdf/1605.06409.pdf)                | [NIPS2016] R-FCN: Object Detection via Region-based Fully Convolutional Networks | ![12](../../../assets/images/docs_images/12.png)             |
|      | [R-FCN++](https://pdfs.semanticscholar.org/f4a2/732d4051b9c4b5d1f057aaa7935be390f51e.pdf) | [AAAI2018] R-FCN++: Towards Accurate Region-Based Fully Convolutional Networks for Object Detection | ![13](../../../assets/images/docs_images/13.png)             |
| V    | [Faster RCNN](https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf) | [NIPS2017] Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks | ![10](../../../assets/images/docs_images/10.png)             |
| V    | [Mask RCNN](https://arxiv.org/pdf/1703.06870.pdf)            | [ICCV2017] Mask R-CNN                                        | The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps<br />![09](../../../assets/images/docs_images/09.png) |


## Relation Network

| V    | Model                                                        | Paper                                                        | Note                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
|      | [Paper](https://arxiv.org/pdf/1711.11575.pdf?fbclid=IwAR2SKIuG2_Izg7BNl6vhXBlAhkwEVxC2yt0ToP2R2CVg6IFKRqmqa-xd3C4) | [CVPR2018]Relation Networks for Object Detection             | This work proposes an object relation module. It processes a set of objects simultaneously through interaction between their appearance feature and geometry, thus allowing modeling of their relations |
| V    | [Reasoning-RCNN](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Reasoning-RCNN_Unifying_Adaptive_Global_Reasoning_Into_Large-Scale_Object_Detection_CVPR_2019_paper.pdf) | [CVPR2019]Reasoning-RCNN: Unifying Adaptive Global Reasoning into Large-scale Object Detection | ![03](../../../assets/images/docs_images/03.png)             |
|      | [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Spatial-Aware_Graph_Relation_Network_for_Large-Scale_Object_Detection_CVPR_2019_paper.pdf) | [CVPR2019] Spatial-aware Graph Relation Network for Large-scale Object Detection | ![14](../../../assets/images/docs_images/14.png)             |

## Tiny things

| V    | Model                                                        | Paper                                                        | Note                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
|      | [FPN](https://arxiv.org/pdf/1612.03144.pdf)     | [CVPR2017]Feature Pyramid Networks for Object Detection      |                                                              |
|      | [MDSSD](https://arxiv.org/pdf/1805.07009v2.pdf) | [2018]MDSSD: Multi-scale Deconvolutional Single Shot Detector for Small Objects | ![04](../../../assets/images/docs_images/04.png)             |
|      | [Paper](https://arxiv.org/pdf/1902.07296.pdf)   | [2019]Augmentation for small object detection                | We conjecture this is due to two factors; (1) only a few images are containing small objects, and (2) small objects do not appear enough even within each image containing them. We thus propose to oversample those images with small objects and augment each of those images by copy-pasting small objects many times |
|      | [Paper](https://arxiv.org/pdf/1901.01892.pdf)   | [ICCV2019] Scale-Aware Trident Networks for Object Detection | ![01](../../../assets/images/docs_images/01.png)             |
|      | [Paper](https://arxiv.org/pdf/1902.06042.pdf)   | [2019]R^2 -CNN: Fast Tiny Object Detection in Large-scale Remote Sensing Images |                                                              |


## RPN

| V    | Model                                                        | Paper                                                        | Note                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
|      | [Faster-RCNN](https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks) | [NIPS2015]Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks |                                                              |
|      | [Anchor-free RPN](https://arxiv.org/abs/1804.09003)          | [2018] An Anchor-Free Region Proposal Network for Faster R-CNN based Text Detection Approaches | In order to better enclose scene text instances of various shapes, it requires to design anchors of various scales, aspect ratios and even orientations manually, which makes anchorbased methods sophisticated and inefficient<br />Compared with a vanilla RPN and FPN-RPN, AF-RPN can get rid of complicated anchor design and achieve higher recall rate on large-scale COCO-Text dataset. |
|      | [CRAFT](https://arxiv.org/pdf/1604.03239.pdf)                | [CVPR2016] CRAFT Objects from Images                         | ![08](../../../assets/images/docs_images/08.png)             |
|      | [Siamese RPN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf) | [CVPR2018] High Performance Visual Tracking with Siamese Region Proposal Network | ![11](../../../assets/images/docs_images/11.png)             |
|      | [Siamese cRPN](https://zpascal.net/cvpr2019/Fan_Siamese_Cascaded_Region_Proposal_Networks_for_Real-Time_Visual_Tracking_CVPR_2019_paper.pdf) | [CVPR2019] Siamese Cascaded Region Proposal Networks for **Real-Time Visual Tracking** | previously proposed one-stage Siamese-RPN trackers degenerate in presence of similar distractors and large scale variation. Addressing these issues, we propose a multi-stage tracking framework, Siamese Cascaded RPN (C-RPN), which consists of a sequence of RPNs cascaded from deep high-level to shallow low-level layers in a Siamese network. |
|      | [Enhanced-PRN](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0203897&type=printable) | [PLOS2018] An Enhanced Region Proposal Network for object detection using deep learning method |                                                              |
|      | [GA-Nets](https://arxiv.org/pdf/1901.03278.pdf)              | [CVPR2019] Region Proposal by Guided Anchoring               |                                                              |