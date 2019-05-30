# deep learning object detection
A paper list of object detection using deep learning. I wrote this page with reference to [this survey paper](https://arxiv.org/pdf/1809.02165v1.pdf) and searching and searching.. 

*Last updated: 2019/05/23*

#### Update log
*2018/9/18* - update all of recent papers and make some diagram about history of object detection using deep learning. 
*2018/9/26* - update codes of papers. (official and unofficial)  
*2018/october* - update 5 papers and performance table.  
*2018/november* - update 9 papers.  
*2018/december* - update 8 papers and and performance table and add new diagram(**2019 version!!**).  
*2019/january* - update 4 papers and and add commonly used datasets.  
*2019/february* - update 3 papers.  
*2019/march* - update figure and code links.  
*2019/april* - remove author's names and update ICLR 2019 & CVPR 2019 papers.  
*2019/may* - update CVPR 2019 papers.  


##

## Table of Contents
- [Paper list from 2014 to now(2019)](https://github.com/hoya012/deep_learning_object_detection/blob/master/README.md#paper-list-from-2014-to-now2019)
- [Performance table](https://github.com/hoya012/deep_learning_object_detection/blob/master/README.md#performance-table)
- Papers
  - [2014](https://github.com/hoya012/deep_learning_object_detection/blob/master/README.md#2014)
  - [2015](https://github.com/hoya012/deep_learning_object_detection/blob/master/README.md#2015)
  - [2016](https://github.com/hoya012/deep_learning_object_detection/blob/master/README.md#2016)
  - [2017](https://github.com/hoya012/deep_learning_object_detection/blob/master/README.md#2017)
  - [2018](https://github.com/hoya012/deep_learning_object_detection/blob/master/README.md#2018)
  - [2019](https://github.com/hoya012/deep_learning_object_detection/blob/master/README.md#2019)
- [Dataset Papers](https://github.com/hoya012/deep_learning_object_detection/blob/master/README.md#dataset-papers)

##

## Paper list from 2014 to now(2019)
The part highlighted with red characters means papers that i think "must-read".
However, it is **my personal opinion** and other papers are important too, so I recommend to read them if you have time.

<p align="center">
  <img width="1000" src="/assets/deep_learning_object_detection_history.PNG" "Example of object detection.">
</p>

##

## Performance table

FPS(Speed) index is related to the hardware spec(e.g. CPU, GPU, RAM, etc), so it is hard to make an equal comparison. The solution is to measure the performance of all models on hardware with equivalent specifications, but it is very difficult and time consuming. 

|   Detector   | VOC07 (mAP@IoU=0.5) | VOC12 (mAP@IoU=0.5) | COCO (mAP@IoU=0.5:0.95) | Published In |
|:------------:|:-------------------:|:-------------------:|:----------:|:------------:| 
|     R-CNN    |         58.5        |          -          |      -     |    CVPR'14   |
|    SPP-Net   |         59.2        |          -          |      -     |    ECCV'14   |
|    MR-CNN    |     78.2 (07+12)    |     73.9 (07+12)    |      -     |    ICCV'15   |
|  Fast R-CNN  |     70.0 (07+12)    |     68.4 (07++12)   |    19.7    |    ICCV'15   |
| Faster R-CNN |     73.2 (07+12)    |     70.4 (07++12)   |    21.9    |    NIPS'15   |
|    YOLO v1   |     66.4 (07+12)    |     57.9 (07++12)   |      -     |    CVPR'16   |
|     G-CNN    |         66.8        |     66.4 (07+12)    |      -     |    CVPR'16   |
|     AZNet    |         70.4        |          -          |    22.3    |    CVPR'16   |
|      ION     |         80.1        |         77.9        |    33.1    |    CVPR'16   |
|   HyperNet   |     76.3 (07+12)    |    71.4 (07++12)    |      -     |    CVPR'16   |
|     OHEM     |     78.9 (07+12)    |    76.3 (07++12)    |    22.4    |    CVPR'16   |
|      MPN     |           -         |          -          |    33.2    |    BMVC'16   |
|      SSD     |     76.8 (07+12)    |    74.9 (07++12)    |    31.2    |    ECCV'16   |
|    GBDNet    |     77.2 (07+12)    |          -          |    27.0    |    ECCV'16   |
|      CPF     |     76.4 (07+12)    |    72.6 (07++12)    |      -     |    ECCV'16   |
|     R-FCN    |     79.5 (07+12)    |    77.6 (07++12)    |    29.9    |    NIPS'16   |
|  DeepID-Net  |         69.0        |          -          |      -     |    PAMI'16   |
|      NoC     |     71.6 (07+12)    |    68.8 (07+12)     |    27.2    |   TPAMI'16   |
|     DSSD     |     81.5 (07+12)    |    80.0 (07++12)    |    33.2    |   arXiv'17   |
|      TDM     |          -          |          -          |    37.3    |    CVPR'17   |
|      FPN     |          -          |          -          |    36.2    |    CVPR'17   |
|    YOLO v2   |     78.6 (07+12)    |    73.4 (07++12)    |      -     |    CVPR'17   |
|      RON     |     77.6 (07+12)    |    75.4 (07++12)    |    27.4    |    CVPR'17   |
|     DeNet    |     77.1 (07+12)    |    73.9 (07++12)    |    33.8    |    ICCV'17   |
|   CoupleNet  |     82.7 (07+12)    |    80.4 (07++12)    |    34.4    |    ICCV'17   |
|   RetinaNet  |          -          |          -          |    39.1    |    ICCV'17   |
|     DSOD     |     77.7 (07+12)    |    76.3 (07++12)    |      -     |    ICCV'17   |
|      SMN     |         70.0        |          -          |      -     |    ICCV'17   |
|Light-Head R-CNN|        -          |          -          |    41.5    |   arXiv'17   |
|    YOLO v3   |          -          |          -          |    33.0    |   arXiv'18   |
|      SIN     |     76.0 (07+12)    |    73.1 (07++12)    |    23.2    |    CVPR'18   |
|     STDN     |     80.9 (07+12)    |          -          |      -     |    CVPR'18   |
|   RefineDet  |     83.8 (07+12)    |    83.5 (07++12)    |    41.8    |    CVPR'18   |
|     SNIP     |          -          |          -          |  **45.7**  |    CVPR'18   |
|Relation-Network|        -          |          -          |     32.5   |    CVPR'18   |
| Cascade R-CNN|          -          |          -          |     42.8   |    CVPR'18   |
|     MLKP     |     80.6 (07+12)    |    77.2 (07++12)    |     28.6   |    CVPR'18   |
|  Fitness-NMS |          -          |          -          |     41.8   |    CVPR'18   |
|    RFBNet    |     82.2 (07+12)    |          -          |      -     |    ECCV'18   |
|   CornerNet  |          -          |          -          |     42.1   |    ECCV'18   |
|    PFPNet    |   **84.1 (07+12)**  |  **83.7 (07++12)**  |     39.4   |    ECCV'18   |
|    Pelee     |     70.9 (07+12)    |          -          |      -     |    NIPS'18   |
|     HKRM     |     78.8 (07+12)    |          -          |     37.8   |    NIPS'18   |
|     M2Det    |          -          |          -          |     44.2   |    AAAI'19   |
|     R-DAD    |     81.2 (07++12)   |    82.0 (07++12)    |     43.1   |    AAAI'19   |

##

## 2014

- **[R-CNN]** Rich feature hierarchies for accurate object detection and semantic segmentation | **[CVPR' 14]** |[`[pdf]`](https://arxiv.org/pdf/1311.2524.pdf) [`[official code - caffe]`](https://github.com/rbgirshick/rcnn) 

- **[OverFeat]** OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks | **[ICLR' 14]** |[`[pdf]`](https://arxiv.org/pdf/1312.6229.pdf) [`[official code - torch]`](https://github.com/sermanet/OverFeat) 

- **[MultiBox]** Scalable Object Detection using Deep Neural Networks | **[CVPR' 14]** |[`[pdf]`](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Erhan_Scalable_Object_Detection_2014_CVPR_paper.pdf)

- **[SPP-Net]** Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition | **[ECCV' 14]** |[`[pdf]`](https://arxiv.org/pdf/1406.4729.pdf) [`[official code - caffe]`](https://github.com/ShaoqingRen/SPP_net) [`[unofficial code - keras]`](https://github.com/yhenon/keras-spp) [`[unofficial code - tensorflow]`](https://github.com/peace195/sppnet)

## 2015
- Improving Object Detection with Deep Convolutional Networks via Bayesian Optimization and Structured Prediction | **[CVPR' 15]** |[`[pdf]`](https://arxiv.org/pdf/1504.03293.pdf) [`[official code - matlab]`](https://github.com/YutingZhang/fgs-obj)

- **[MR-CNN]** Object detection via a multi-region & semantic segmentation-aware CNN model | **[ICCV' 15]** |[`[pdf]`](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Gidaris_Object_Detection_via_ICCV_2015_paper.pdf) [`[official code - caffe]`](https://github.com/gidariss/mrcnn-object-detection)

- **[DeepBox]** DeepBox: Learning Objectness with Convolutional Networks | **[ICCV' 15]** |[`[pdf]`](https://arxiv.org/pdf/1505.02146.pdf) [`[official code - caffe]`](https://github.com/weichengkuo/DeepBox)

- **[AttentionNet]** AttentionNet: Aggregating Weak Directions for Accurate Object Detection | **[ICCV' 15]** |[`[pdf]`](https://arxiv.org/pdf/1506.07704.pdf) 

- **[Fast R-CNN]** Fast R-CNN | **[ICCV' 15]** |[`[pdf]`](https://arxiv.org/pdf/1504.08083.pdf) [`[official code - caffe]`](https://github.com/rbgirshick/fast-rcnn) 

- **[DeepProposal]** DeepProposal: Hunting Objects by Cascading Deep Convolutional Layers | **[ICCV' 15]** |[`[pdf]`](https://arxiv.org/pdf/1510.04445.pdf)  [`[official code - matconvnet]`](https://github.com/aghodrati/deepproposal)

- **[Faster R-CNN, RPN]** Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks | **[NIPS' 15]** |[`[pdf]`](https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf)  [`[official code - caffe]`](https://github.com/rbgirshick/py-faster-rcnn) [`[unofficial code - tensorflow]`](https://github.com/endernewton/tf-faster-rcnn)  [`[unofficial code - pytorch]`](https://github.com/jwyang/faster-rcnn.pytorch) 

## 2016

- **[YOLO v1]** You Only Look Once: Unified, Real-Time Object Detection | **[CVPR' 16]** |[`[pdf]`](https://arxiv.org/pdf/1506.02640.pdf) [`[official code - c]`](https://pjreddie.com/darknet/yolo/) 

- **[G-CNN]** G-CNN: an Iterative Grid Based Object Detector | **[CVPR' 16]** |[`[pdf]`](https://arxiv.org/pdf/1512.07729.pdf)

- **[AZNet]** Adaptive Object Detection Using Adjacency and Zoom Prediction | **[CVPR' 16]** |[`[pdf]`](https://arxiv.org/pdf/1512.07711.pdf)

- **[ION]** Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks | **[CVPR' 16]** |[`[pdf]`](https://arxiv.org/pdf/1512.04143.pdf)

- **[HyperNet]** HyperNet: Towards Accurate Region Proposal Generation and Joint Object Detection | **[CVPR' 16]** |[`[pdf]`](https://arxiv.org/pdf/1604.00600.pdf)

- **[OHEM]** Training Region-based Object Detectors with Online Hard Example Mining | **[CVPR' 16]** |[`[pdf]`](https://arxiv.org/pdf/1604.03540.pdf) [`[official code - caffe]`](https://github.com/abhi2610/ohem) 

- **[CRAPF]** CRAFT Objects from Images | **[CVPR' 16]** |[`[pdf]`](https://arxiv.org/pdf/1604.03239.pdf) [`[official code - caffe]`](https://github.com/byangderek/CRAFT) 

- **[MPN]** A MultiPath Network for Object Detection | **[BMVC' 16]** |[`[pdf]`](https://arxiv.org/pdf/1604.02135.pdf) [`[official code - torch]`](https://github.com/facebookresearch/multipathnet) 

- **[SSD]** SSD: Single Shot MultiBox Detector | **[ECCV' 16]** |[`[pdf]`](https://arxiv.org/pdf/1512.02325.pdf) [`[official code - caffe]`](https://github.com/weiliu89/caffe/tree/ssd) [`[unofficial code - tensorflow]`](https://github.com/balancap/SSD-Tensorflow) [`[unofficial code - pytorch]`](https://github.com/amdegroot/ssd.pytorch) 

- **[GBDNet]** Crafting GBD-Net for Object Detection | **[ECCV' 16]** |[`[pdf]`](https://arxiv.org/pdf/1610.02579.pdf) [`[official code - caffe]`](https://github.com/craftGBD/craftGBD)

- **[CPF]** Contextual Priming and Feedback for Faster R-CNN | **[ECCV' 16]** |[`[pdf]`](https://pdfs.semanticscholar.org/40e7/4473cb82231559cbaeaa44989e9bbfe7ec3f.pdf)

- **[MS-CNN]** A Unified Multi-scale Deep Convolutional Neural Network for Fast Object Detection | **[ECCV' 16]** |[`[pdf]`](https://arxiv.org/pdf/1607.07155.pdf) [`[official code - caffe]`](https://github.com/zhaoweicai/mscnn)

- **[R-FCN]** R-FCN: Object Detection via Region-based Fully Convolutional Networks | **[NIPS' 16]** |[`[pdf]`](https://arxiv.org/pdf/1605.06409.pdf) [`[official code - caffe]`](https://github.com/daijifeng001/R-FCN) [`[unofficial code - caffe]`](https://github.com/YuwenXiong/py-R-FCN)

- **[PVANET]** PVANET: Deep but Lightweight Neural Networks for Real-time Object Detection | **[NIPSW' 16]** |[`[pdf]`](https://arxiv.org/pdf/1608.08021.pdf) [`[official code - caffe]`](https://github.com/sanghoon/pva-faster-rcnn)

- **[DeepID-Net]** DeepID-Net: Deformable Deep Convolutional Neural Networks for Object Detection | **[PAMI' 16]** |[`[pdf]`](https://arxiv.org/pdf/1412.5661.pdf)

- **[NoC]** Object Detection Networks on Convolutional Feature Maps | **[TPAMI' 16]** |[`[pdf]`](https://arxiv.org/pdf/1504.06066.pdf)

## 2017

- **[DSSD]** DSSD : Deconvolutional Single Shot Detector | **[arXiv' 17]** |[`[pdf]`](https://arxiv.org/pdf/1701.06659.pdf) [`[official code - caffe]`](https://github.com/chengyangfu/caffe/tree/dssd)

- **[TDM]** Beyond Skip Connections: Top-Down Modulation for Object Detection | **[CVPR' 17]** |[`[pdf]`](https://arxiv.org/pdf/1612.06851.pdf)

- **[FPN]** Feature Pyramid Networks for Object Detection  | **[CVPR' 17]** |[`[pdf]`](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf) [`[unofficial code - caffe]`](https://github.com/unsky/FPN)

- **[YOLO v2]** YOLO9000: Better, Faster, Stronger | **[CVPR' 17]** |[`[pdf]`](https://arxiv.org/pdf/1612.08242.pdf) [`[official code - c]`](https://pjreddie.com/darknet/yolo/) [`[unofficial code - caffe]`](https://github.com/quhezheng/caffe_yolo_v2) [`[unofficial code - tensorflow]`](https://github.com/nilboy/tensorflow-yolo) [`[unofficial code - tensorflow]`](https://github.com/sualab/object-detection-yolov2) [`[unofficial code - pytorch]`](https://github.com/longcw/yolo2-pytorch) 

- **[RON]** RON: Reverse Connection with Objectness Prior Networks for Object Detection | **[CVPR' 17]** |[`[pdf]`](https://arxiv.org/pdf/1707.01691.pdf) [`[official code - caffe]`](https://github.com/taokong/RON) [`[unofficial code - tensorflow]`](https://github.com/HiKapok/RON_Tensorflow)

- **[RSA]** Recurrent Scale Approximation for Object Detection in CNN |  | **[ICCV' 17]** |[`[pdf]`](https://arxiv.org/pdf/1707.09531.pdf) [`[official code - caffe]`](https://github.com/sciencefans/RSA-for-object-detection)

- **[DCN]** Deformable Convolutional Networks  | **[ICCV' 17]** |[`[pdf]`](http://openaccess.thecvf.com/content_ICCV_2017/papers/Dai_Deformable_Convolutional_Networks_ICCV_2017_paper.pdf) [`[official code - mxnet]`](https://github.com/msracver/Deformable-ConvNets) [`[unofficial code - tensorflow]`](https://github.com/Zardinality/TF_Deformable_Net) [`[unofficial code - pytorch]`](https://github.com/oeway/pytorch-deform-conv)

- **[DeNet]** DeNet: Scalable Real-time Object Detection with Directed Sparse Sampling | **[ICCV' 17]** |[`[pdf]`](https://arxiv.org/pdf/1703.10295.pdf) [`[official code - theano]`](https://github.com/lachlants/denet)

- **[CoupleNet]** CoupleNet: Coupling Global Structure with Local Parts for Object Detection | **[ICCV' 17]** |[`[pdf]`](https://arxiv.org/pdf/1708.02863.pdf) [`[official code - caffe]`](https://github.com/tshizys/CoupleNet)

- **[RetinaNet]** Focal Loss for Dense Object Detection | **[ICCV' 17]** |[`[pdf]`](https://arxiv.org/pdf/1708.02002.pdf) [`[official code - keras]`](https://github.com/fizyr/keras-retinanet) [`[unofficial code - pytorch]`](https://github.com/kuangliu/pytorch-retinanet) [`[unofficial code - mxnet]`](https://github.com/unsky/RetinaNet) [`[unofficial code - tensorflow]`](https://github.com/tensorflow/tpu/tree/master/models/official/retinanet)

- **[Mask R-CNN]** Mask R-CNN | **[ICCV' 17]** |[`[pdf]`](http://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf) [`[official code - caffe2]`](https://github.com/facebookresearch/Detectron) [`[unofficial code - tensorflow]`](https://github.com/matterport/Mask_RCNN) [`[unofficial code - tensorflow]`](https://github.com/CharlesShang/FastMaskRCNN) [`[unofficial code - pytorch]`](https://github.com/multimodallearning/pytorch-mask-rcnn)

- **[DSOD]** DSOD: Learning Deeply Supervised Object Detectors from Scratch | **[ICCV' 17]** |[`[pdf]`](https://arxiv.org/pdf/1708.01241.pdf) [`[official code - caffe]`](https://github.com/szq0214/DSOD) [`[unofficial code - pytorch]`](https://github.com/uoip/SSD-variants) 

- **[SMN]** Spatial Memory for Context Reasoning in Object Detection | **[ICCV' 17]** |[`[pdf]`](http://openaccess.thecvf.com/content_ICCV_2017/papers/Chen_Spatial_Memory_for_ICCV_2017_paper.pdf)

- **[Light-Head R-CNN]** Light-Head R-CNN: In Defense of Two-Stage Object Detector | **[arXiv' 17]** |[`[pdf]`](https://arxiv.org/pdf/1711.07264.pdf) [`[official code - tensorflow]`](https://github.com/zengarden/light_head_rcnn)

- **[Soft-NMS]** Improving Object Detection With One Line of Code | **[ICCV' 17]** |[`[pdf]`](https://arxiv.org/pdf/1704.04503.pdf) [`[official code - caffe]`](https://github.com/bharatsingh430/soft-nms)

## 2018

- **[YOLO v3]** YOLOv3: An Incremental Improvement | **[arXiv' 18]** |[`[pdf]`](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [`[official code - c]`](https://pjreddie.com/darknet/yolo/) [`[unofficial code - pytorch]`](https://github.com/ayooshkathuria/pytorch-yolo-v3) [`[unofficial code - pytorch]`](https://github.com/eriklindernoren/PyTorch-YOLOv3) [`[unofficial code - keras]`](https://github.com/qqwweee/keras-yolo3) [`[unofficial code - tensorflow]`](https://github.com/mystic123/tensorflow-yolo-v3)

- **[ZIP]** Zoom Out-and-In Network with Recursive Training for Object Proposal | **[IJCV' 18]** |[`[pdf]`](https://arxiv.org/pdf/1702.05711.pdf) [`[official code - caffe]`](https://github.com/hli2020/zoom_network)

- **[SIN]** Structure Inference Net: Object Detection Using Scene-Level Context and Instance-Level Relationships | **[CVPR' 18]** |[`[pdf]`](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Structure_Inference_Net_CVPR_2018_paper.pdf) [`[official code - tensorflow]`](https://github.com/choasup/SIN)

- **[STDN]** Scale-Transferrable Object Detection | **[CVPR' 18]** |[`[pdf]`](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_Scale-Transferrable_Object_Detection_CVPR_2018_paper.pdf)

- **[RefineDet]** Single-Shot Refinement Neural Network for Object Detection | **[CVPR' 18]** |[`[pdf]`](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single-Shot_Refinement_Neural_CVPR_2018_paper.pdf) [`[official code - caffe]`](https://github.com/sfzhang15/RefineDet) [`[unofficial code - chainer]`](https://github.com/fukatani/RefineDet_chainer)  [`[unofficial code - pytorch]`](https://github.com/lzx1413/PytorchSSD)

- **[MegDet]** MegDet: A Large Mini-Batch Object Detector | **[CVPR' 18]** |[`[pdf]`](http://openaccess.thecvf.com/content_cvpr_2018/papers/Peng_MegDet_A_Large_CVPR_2018_paper.pdf)

- **[DA Faster R-CNN]** Domain Adaptive Faster R-CNN for Object Detection in the Wild | **[CVPR' 18]** |[`[pdf]`](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Domain_Adaptive_Faster_CVPR_2018_paper.pdf) [`[official code - caffe]`](https://github.com/yuhuayc/da-faster-rcnn)

- **[SNIP]** An Analysis of Scale Invariance in Object Detection – SNIP | **[CVPR' 18]** |[`[pdf]`](https://arxiv.org/pdf/1711.08189.pdf)

- **[Relation-Network]** Relation Networks for Object Detection | **[CVPR' 18]** |[`[pdf]`](https://arxiv.org/pdf/1711.11575.pdf) [`[official code - mxnet]`](https://github.com/msracver/Relation-Networks-for-Object-Detection)

- **[Cascade R-CNN]** Cascade R-CNN: Delving into High Quality Object Detection | **[CVPR' 18]** |[`[pdf]`](http://openaccess.thecvf.com/content_cvpr_2018/papers/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.pdf) [`[official code - caffe]`](https://github.com/zhaoweicai/cascade-rcnn)

- Finding Tiny Faces in the Wild with Generative Adversarial Network | **[CVPR' 18]** |[`[pdf]`](https://ivul.kaust.edu.sa/Documents/Publications/2018/Finding%20Tiny%20Faces%20in%20the%20Wild%20with%20Generative%20Adversarial%20Network.pdf)

- **[MLKP]** Multi-scale Location-aware Kernel Representation for Object Detection | **[CVPR' 18]** |[`[pdf]`](https://arxiv.org/pdf/1804.00428.pdf) [`[official code - caffe]`](https://github.com/Hwang64/MLKP)

- Cross-Domain Weakly-Supervised Object Detection through Progressive Domain Adaptation | **[CVPR' 18]** |[`[pdf]`](https://arxiv.org/pdf/1803.11365.pdf) [`[official code - chainer]`](https://github.com/naoto0804/cross-domain-detection)

- **[Fitness NMS]** Improving Object Localization with Fitness NMS and Bounded IoU Loss | **[CVPR' 18]** |[`[pdf]`](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0794.pdf) 

- **[STDnet]** STDnet: A ConvNet for Small Target Detection | **[BMVC' 18]** |[`[pdf]`](http://bmvc2018.org/contents/papers/0897.pdf)

- **[RFBNet]** Receptive Field Block Net for Accurate and Fast Object Detection | **[ECCV' 18]** |[`[pdf]`](https://arxiv.org/pdf/1711.07767.pdf) [`[official code - pytorch]`](https://github.com/ruinmessi/RFBNet)

- Zero-Annotation Object Detection with Web Knowledge Transfer | **[ECCV' 18]** |[`[pdf]`](http://openaccess.thecvf.com/content_ECCV_2018/papers/Qingyi_Tao_Zero-Annotation_Object_Detection_ECCV_2018_paper.pdf)

- **[CornerNet]** CornerNet: Detecting Objects as Paired Keypoints | **[ECCV' 18]** |[`[pdf]`](https://arxiv.org/pdf/1808.01244.pdf) [`[official code - pytorch]`](https://github.com/princeton-vl/CornerNet)

- **[PFPNet]** Parallel Feature Pyramid Network for Object Detection | **[ECCV' 18]** |[`[pdf]`](http://openaccess.thecvf.com/content_ECCV_2018/papers/Seung-Wook_Kim_Parallel_Feature_Pyramid_ECCV_2018_paper.pdf)

- **[Softer-NMS]** Softer-NMS: Rethinking Bounding Box Regression for Accurate Object Detection | **[arXiv' 18]** |[`[pdf]`](https://arxiv.org/pdf/1809.08545.pdf)

- **[ShapeShifter]** ShapeShifter: Robust Physical Adversarial Attack on Faster R-CNN Object Detector | **[ECML-PKDD' 18]** |[`[pdf]`](https://arxiv.org/pdf/1804.05810.pdf) [`[official code - tensorflow]`](https://github.com/shangtse/robust-physical-attack)

- **[Pelee]** Pelee: A Real-Time Object Detection System on Mobile Devices | **[NIPS' 18]** |[`[pdf]`](http://papers.nips.cc/paper/7466-pelee-a-real-time-object-detection-system-on-mobile-devices.pdf) [`[official code - caffe]`](https://github.com/Robert-JunWang/Pelee)

- **[HKRM]** Hybrid Knowledge Routed Modules for Large-scale Object Detection | **[NIPS' 18]** |[`[pdf]`](http://papers.nips.cc/paper/7428-hybrid-knowledge-routed-modules-for-large-scale-object-detection.pdf) 

- **[MetaAnchor]** MetaAnchor: Learning to Detect Objects with Customized Anchors | **[NIPS' 18]** |[`[pdf]`](http://papers.nips.cc/paper/7315-metaanchor-learning-to-detect-objects-with-customized-anchors.pdf) 

- **[SNIPER]** SNIPER: Efficient Multi-Scale Training | **[NIPS' 18]** |[`[pdf]`](http://papers.nips.cc/paper/8143-sniper-efficient-multi-scale-training.pdf) 

## 2019
- **[M2Det]** M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network | **[AAAI' 19]** |[`[pdf]`](https://arxiv.org/pdf/1811.04533.pdf) [`[official code - pytorch]`](https://github.com/qijiezhao/M2Det)

- **[R-DAD]** Object Detection based on Region Decomposition and Assembly | **[AAAI' 19]** |[`[pdf]`](https://arxiv.org/pdf/1901.08225v1.pdf) 

- **[CAMOU]** CAMOU: Learning Physical Vehicle Camouflages to Adversarially Attack Detectors in the Wild | **[ICLR' 19]** |[`[pdf]`](https://openreview.net/pdf?id=SJgEl3A5tm) 

- Feature Intertwiner for Object Detection | **[ICLR' 19]** |[`[pdf]`](https://openreview.net/pdf?id=SyxZJn05YX) 

- **[GIoU]** Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1902.09630.pdf) 

- Automatic adaptation of object detectors to new domains using self-training | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.07305.pdf) 

- **[Libra R-CNN]** Libra R-CNN: Balanced Learning for Object Detection | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.02701.pdf) 

- Feature Selective Anchor-Free Module for Single-Shot Object Detection | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1903.00621.pdf) 

- **[ExtremeNet]** Bottom-up Object Detection by Grouping Extreme and Center Points | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1901.08043.pdf) | [`[official code - pytorch]`](https://github.com/xingyizhou/ExtremeNet)

- **[C-MIL]** C-MIL: Continuation Multiple Instance Learning for Weakly Supervised Object Detection
 | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.05647.pdf) | [`[official code - torch]`](https://github.com/AnonymousIDs/C-MIL)

- **[ScratchDet]** ScratchDet: Training Single-Shot Object Detectors from Scratch | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1810.08425.pdf) 

- Bounding Box Regression with Uncertainty for Accurate Object Detection | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1809.08545.pdf) | [`[official code - caffe2]`](https://github.com/yihui-he/KL-Loss)

- Activity Driven Weakly Supervised Object Detection | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.01665.pdf) | [`[official code - pytorch]`](https://github.com/zhenheny/ADWSOD) 

- Towards Accurate One-Stage Object Detection with AP-Loss | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.06373.pdf) 

- Strong-Weak Distribution Alignment for Adaptive Object Detection | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1812.04798.pdf) | [`[official code - pytorch]`](https://github.com/VisionLearningGroup/DA_Detection) 

- **[NAS-FPN]** NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.07392.pdf) 

- **[Adaptive NMS]** Adaptive NMS: Refining Pedestrian Detection in a Crowd | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.03629.pdf) 

- Point in, Box out: Beyond Counting Persons in Crowds | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.01333.pdf) 

- Locating Objects Without Bounding Boxes | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1806.07564.pdf) 

- Sampling Techniques for Large-Scale Object Detection from Sparsely Annotated Objects | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1811.10862.pdf) 

- Towards Universal Object Detection by Domain Attention | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.04402.pdf) 

- Exploring the Bounds of the Utility of Context for Object Detection | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1711.05471.pdf) 

- What Object Should I Use? - Task Driven Object Detection | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.03000.pdf) 

- Dissimilarity Coefficient based Weakly Supervised Object Detection | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1811.10016) 

- Adapting Object Detectors via Selective Cross-Domain Alignment | **[CVPR' 19]**

- Fully Quantized Network for Object Detection | **[CVPR' 19]** |[`[pdf]`](https://yan-junjie.github.io/publication/dblp-confcvprlilqwfy-19/dblp-confcvprlilqwfy-19.pdf)

- Distilling Object Detectors with Fine-grained Feature Imitation | **[CVPR' 19]**

- Multi-task Self-Supervised Object Detection via Recycling of Bounding Box Annotations | **[CVPR' 19]**

- **[Reasoning-RCNN]** Reasoning-RCNN: Unifying Adaptive Global Reasoning into Large-scale Object Detection | **[CVPR' 19]**

- Arbitrary Shape Scene Text Detection with Adaptive Text Region Representation | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1905.05980.pdf)

- Assisted Excitation of Activations: A Learning Technique to Improve Object Detectors | **[CVPR' 19]** |[`[pdf]`](https://pdfs.semanticscholar.org/ec96/b6ae95e1ebbe4f7c0252301ede26dfc79467.pdf)

- Spatial-aware Graph Relation Network for Large-scale Object Detection | **[CVPR' 19]**

- **[MaxpoolNMS]** MaxpoolNMS: Getting Rid of NMS Bottlenecks in Two-Stage Object Detectors | **[CVPR' 19]**

- You reap what you sow: Generating High Precision Object Proposals for Weakly-supervised Object Detection | **[CVPR' 19]** |[`[pdf]`](https://web.cs.ucdavis.edu/~yjlee/projects/cvpr2019-youreapwhatyousow.pdf)

- Object detection with location-aware deformable convolution and backward attention filtering | **[CVPR' 19]**

- Diversify and Match: A Domain Adaptive Representation Learning Paradigm for Object Detection | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1905.05396.pdf)

##

## Dataset Papers
Statistics of commonly used object detection datasets. The Figure came from [this survey paper](https://arxiv.org/pdf/1809.02165v1.pdf).
<p align="center">
  <img width="700" src="/assets/deep_learning_object_detection_dataset.PNG" "Commonly used object detection datasets.">
</p>

The papers related to datasets used mainly in Object Detection are as follows.

- **[PASCAL VOC]** The PASCAL Visual Object Classes (VOC) Challenge | **[IJCV' 10]** | [`[pdf]`](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf)

- **[PASCAL VOC]** The PASCAL Visual Object Classes Challenge: A Retrospective | **[IJCV' 15]** | [`[pdf]`](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham15.pdf) | [`[link]`](http://host.robots.ox.ac.uk/pascal/VOC/)

- **[ImageNet]** ImageNet: A Large-Scale Hierarchical Image Database| **[CVPR' 09]** | [`[pdf]`](http://www.image-net.org/papers/imagenet_cvpr09.pdf)

- **[ImageNet]** ImageNet Large Scale Visual Recognition Challenge | **[IJCV' 15]** | [`[pdf]`](https://arxiv.org/pdf/1409.0575.pdf) | [`[link]`](http://www.image-net.org/challenges/LSVRC/)

- **[COCO]** Microsoft COCO: Common Objects in Context | **[ECCV' 14]** | [`[pdf]`](https://arxiv.org/pdf/1405.0312.pdf) | [`[link]`](http://cocodataset.org/)

- **[Open Images]** The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale | **[arXiv' 18]** | [`[pdf]`](https://arxiv.org/pdf/1811.00982v1.pdf) | [`[link]`](https://storage.googleapis.com/openimages/web/index.html)

##

## Contact & Feedback

If you have any suggestions about papers, feel free to mail me :)

- [e-mail](mailto:lee.hoseong@sualab.com)
- [blog](https://hoya012.github.io/)
- [pull request](https://github.com/hoya012/deep_learning_object_detection/pulls)
