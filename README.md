# deep learning object detection
A paper list of object detection using deep learning. I wrote this page with reference to [this survey paper](https://arxiv.org/pdf/1809.02165v1.pdf) and searching and searching.. 

*Last updated: 2020/07/17*

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
*2019/june* - update CVPR 2019 papers and dataset paper.  
*2019/july* - update BMVC 2019 papers and some of ICCV 2019 papers.  
*2019/september* - update NeurIPS 2019 papers and ICCV 2019 papers.  
*2019/november* - update some of AAAI 2020 papers and other papers.  
*2020/january* - update ICLR 2020 papers and other papers.  
*2020/may* - update CVPR 2020 papers and other papers.  
*2020/june* - update arxiv papers.  


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
  - [2020](https://github.com/hoya012/deep_learning_object_detection/blob/master/README.md#2020)
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
|     SNIP     |          -          |          -          |    45.7    |    CVPR'18   |
|Relation-Network|        -          |          -          |     32.5   |    CVPR'18   |
| Cascade R-CNN|          -          |          -          |     42.8   |    CVPR'18   |
|     MLKP     |     80.6 (07+12)    |    77.2 (07++12)    |     28.6   |    CVPR'18   |
|  Fitness-NMS |          -          |          -          |     41.8   |    CVPR'18   |
|    RFBNet    |     82.2 (07+12)    |          -          |      -     |    ECCV'18   |
|   CornerNet  |          -          |          -          |     42.1   |    ECCV'18   |
|    PFPNet    |     84.1 (07+12)    |    83.7 (07++12)    |     39.4   |    ECCV'18   |
|    Pelee     |     70.9 (07+12)    |          -          |      -     |    NIPS'18   |
|     HKRM     |     78.8 (07+12)    |          -          |     37.8   |    NIPS'18   |
|     M2Det    |          -          |          -          |     44.2   |    AAAI'19   |
|     R-DAD    |     81.2 (07++12)   |    82.0 (07++12)    |     43.1   |    AAAI'19   |
| ScratchDet   |   84.1 (07++12)     |    83.6 (07++12)    |     39.1   |    CVPR'19   |
| Libra R-CNN  |          -          |          -          |     43.0   |    CVPR'19   |
| Reasoning-RCNN  | 82.5 (07++12)    |          -          |     43.2   |    CVPR'19   |
|      FSAF    |          -          |          -          |     44.6   |    CVPR'19   |
| AmoebaNet + NAS-FPN |     -        |          -          |     47.0   |    CVPR'19   |
| Cascade-RetinaNet |       -        |           -         |     41.1   |    CVPR'19   |
|      HTC     |          -          |          -          |     47.2   |    CVPR'19   |
|   TridentNet |          -          |          -          |     48.4   |    ICCV'19   |
|      DAFS    |   **85.3 (07+12)**  |    83.1 (07++12)    |     40.5   |    ICCV'19   |
|   Auto-FPN   |     81.8 (07++12)   |          -          |     40.5   |    ICCV'19   |
|     FCOS     |          -          |          -          |     44.7   |    ICCV'19   |
|   FreeAnchor |          -          |          -          |     44.8   |  NeurIPS'19  |
|    DetNAS    |     81.5 (07++12)   |          -          |     42.0   |  NeurIPS'19  |
|     NATS     |          -          |          -          |     42.0   |  NeurIPS'19  |
| AmoebaNet + NAS-FPN + AA |   -     |          -          |     50.7   |    arXiv'19  |
|   SpineNet   |          -          |          -          |     52.1   |    arXiv'19  |
|     CBNet    |          -          |          -          |     53.3   |    AAAI'20   |
| EfficientDet |          -          |          -          |     52.6   |    CVPR'20   |
|  DetectoRS   |          -          |          -          |     **54.7**   |    arXiv'20   |

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

- **[FSAF]** Feature Selective Anchor-Free Module for Single-Shot Object Detection | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1903.00621.pdf) 

- **[ExtremeNet]** Bottom-up Object Detection by Grouping Extreme and Center Points | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1901.08043.pdf) | [`[official code - pytorch]`](https://github.com/xingyizhou/ExtremeNet)

- **[C-MIL]** C-MIL: Continuation Multiple Instance Learning for Weakly Supervised Object Detection
 | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.05647.pdf) | [`[official code - torch]`](https://github.com/AnonymousIDs/C-MIL)

- **[ScratchDet]** ScratchDet: Training Single-Shot Object Detectors from Scratch | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1810.08425.pdf) 

- Bounding Box Regression with Uncertainty for Accurate Object Detection | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1809.08545.pdf) | [`[official code - caffe2]`](https://github.com/yihui-he/KL-Loss)

- Activity Driven Weakly Supervised Object Detection | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.01665.pdf) 

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

- Adapting Object Detectors via Selective Cross-Domain Alignment | **[CVPR' 19]** |[`[pdf]`](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Adapting_Object_Detectors_via_Selective_Cross-Domain_Alignment_CVPR_2019_paper.pdf) 

- Fully Quantized Network for Object Detection | **[CVPR' 19]** |[`[pdf]`](https://yan-junjie.github.io/publication/dblp-confcvprlilqwfy-19/dblp-confcvprlilqwfy-19.pdf)

- Distilling Object Detectors with Fine-grained Feature Imitation | **[CVPR' 19]** |[`[pdf]`](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Distilling_Object_Detectors_With_Fine-Grained_Feature_Imitation_CVPR_2019_paper.pdf)

- Multi-task Self-Supervised Object Detection via Recycling of Bounding Box Annotations | **[CVPR' 19]** |[`[pdf]`](http://openaccess.thecvf.com/content_CVPR_2019/papers/Lee_Multi-Task_Self-Supervised_Object_Detection_via_Recycling_of_Bounding_Box_Annotations_CVPR_2019_paper.pdf)

- **[Reasoning-RCNN]** Reasoning-RCNN: Unifying Adaptive Global Reasoning into Large-scale Object Detection | **[CVPR' 19]** |[`[pdf]`](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Reasoning-RCNN_Unifying_Adaptive_Global_Reasoning_Into_Large-Scale_Object_Detection_CVPR_2019_paper.pdf)

- Arbitrary Shape Scene Text Detection with Adaptive Text Region Representation | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1905.05980.pdf)

- Assisted Excitation of Activations: A Learning Technique to Improve Object Detectors | **[CVPR' 19]** |[`[pdf]`](https://pdfs.semanticscholar.org/ec96/b6ae95e1ebbe4f7c0252301ede26dfc79467.pdf)

- Spatial-aware Graph Relation Network for Large-scale Object Detection | **[CVPR' 19]** |[`[pdf]`](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Spatial-Aware_Graph_Relation_Network_for_Large-Scale_Object_Detection_CVPR_2019_paper.pdf)

- **[MaxpoolNMS]** MaxpoolNMS: Getting Rid of NMS Bottlenecks in Two-Stage Object Detectors | **[CVPR' 19]** |[`[pdf]`](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cai_MaxpoolNMS_Getting_Rid_of_NMS_Bottlenecks_in_Two-Stage_Object_Detectors_CVPR_2019_paper.pdf)

- You reap what you sow: Generating High Precision Object Proposals for Weakly-supervised Object Detection | **[CVPR' 19]** |[`[pdf]`](https://web.cs.ucdavis.edu/~yjlee/projects/cvpr2019-youreapwhatyousow.pdf)

- Object detection with location-aware deformable convolution and backward attention filtering | **[CVPR' 19]** |[`[pdf]`](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Object_Detection_With_Location-Aware_Deformable_Convolution_and_Backward_Attention_Filtering_CVPR_2019_paper.pdf)

- Diversify and Match: A Domain Adaptive Representation Learning Paradigm for Object Detection | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1905.05396.pdf)

- Hybrid Task Cascade for Instance Segmentation | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1901.07518.pdf)

- **[GFR]** Improving Object Detection from Scratch via Gated Feature Reuse | **[BMVC' 19]** |[`[pdf]`](https://arxiv.org/pdf/1712.00886v2.pdf) | [`[official code - pytorch]`](https://github.com/szq0214/GFR-DSOD)

- **[Cascade RetinaNet]** Cascade RetinaNet: Maintaining Consistency for Single-Stage Object Detection | **[BMVC' 19]** |[`[pdf]`](https://arxiv.org/pdf/1907.06881v1.pdf)

- Soft Sampling for Robust Object Detection | **[BMVC' 19]** |[`[pdf]`](https://arxiv.org/pdf/1806.06986v2.pdf)

- Multi-adversarial Faster-RCNN for Unrestricted Object Detection | **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1907.10343v1.pdf)

- Towards Adversarially Robust Object Detection | **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1907.10310v1.pdf)

- A Robust Learning Approach to Domain Adaptive Object Detection | **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.02361.pdf)

- A Delay Metric for Video Object Detection: What Average Precision Fails to Tell	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1908.06368.pdf)

- Delving Into Robust Object Detection From Unmanned Aerial Vehicles: A Deep Nuisance Disentanglement Approach | **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1908.03856.pdf)

- Employing Deep Part-Object Relationships for Salient Object Detection	| **[ICCV' 19]** |[`[pdf]`](http://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Employing_Deep_Part-Object_Relationships_for_Salient_Object_Detection_ICCV_2019_paper.pdf)

- Learning Rich Features at High-Speed for Single-Shot Object Detection	| **[ICCV' 19]** |[`[pdf]`](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Learning_Rich_Features_at_High-Speed_for_Single-Shot_Object_Detection_ICCV_2019_paper.pdf)

- Structured Modeling of Joint Deep Feature and Prediction Refinement for Salient Object Detection	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1909.04366.pdf)

- Selectivity or Invariance: Boundary-Aware Salient Object Detection	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1812.10066.pdf)

- Progressive Sparse Local Attention for Video Object Detection	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1903.09126.pdf)

- Minimum Delay Object Detection From Video	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1908.11092.pdf)

- Towards Interpretable Object Detection by Unfolding Latent Structures	 | **[ICCV' 19]**  |[`[pdf]`](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Towards_Interpretable_Object_Detection_by_Unfolding_Latent_Structures_ICCV_2019_paper.pdf)

- Scaling Object Detection by Transferring Classification Weights	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1909.06804.pdf)

- **[TridentNet]** Scale-Aware Trident Networks for Object Detection	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1901.01892.pdf)

- Generative Modeling for Small-Data Object Detection	| **[ICCV' 19]** |[`[pdf]`](http://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Generative_Modeling_for_Small-Data_Object_Detection_ICCV_2019_paper.pdf)

- Transductive Learning for Zero-Shot Object Detection	| **[ICCV' 19]** |[`[pdf]`](https://salman-h-khan.github.io/papers/ICCV19-2.pdf)

- Self-Training and Adversarial Background Regularization for Unsupervised Domain Adaptive One-Stage Object Detection	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1909.00597.pdf)

-  **[CenterNet]** CenterNet: Keypoint Triplets for Object Detection	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.08189.pdf)

- **[DAFS]** Dynamic Anchor Feature Selection for Single-Shot Object Detection	| **[ICCV' 19]** |[`[pdf]`](http://www4.comp.polyu.edu.hk/~cslzhang/paper/ICCV-DAFS.pdf)

- **[Auto-FPN]** Auto-FPN: Automatic Network Architecture Adaptation for Object Detection Beyond Classification	| **[ICCV' 19]** |[`[pdf]`](http://openaccess.thecvf.com/content_ICCV_2019/papers/Xu_Auto-FPN_Automatic_Network_Architecture_Adaptation_for_Object_Detection_Beyond_Classification_ICCV_2019_paper.pdf)

- Multi-Adversarial Faster-RCNN for Unrestricted Object Detection	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1907.10343.pdf)

- Object Guided External Memory Network for Video Object Detection | **[ICCV' 19]** |[`[pdf]`](http://openaccess.thecvf.com/content_ICCV_2019/papers/Deng_Object_Guided_External_Memory_Network_for_Video_Object_Detection_ICCV_2019_paper.pdf)

- **[ThunderNet]** ThunderNet: Towards Real-Time Generic Object Detection on Mobile Devices	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1903.11752.pdf)

- **[RDN]** Relation Distillation Networks for Video Object Detection	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1908.09511.pdf)

- **[MMNet]** Fast Object Detection in Compressed Video	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1811.11057.pdf)

- Towards High-Resolution Salient Object Detection	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1908.07274.pdf)

- **[SCAN]** Stacked Cross Refinement Network for Edge-Aware Salient Object Detection	| **[ICCV' 19]** |[`[official code]`](https://github.com/wuzhe71/SCAN) |[`[pdf]`]()

- Motion Guided Attention for Video Salient Object Detection	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1909.07061.pdf)

- Semi-Supervised Video Salient Object Detection Using Pseudo-Labels	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1908.04051.pdf)

- Learning to Rank Proposals for Object Detection	| **[ICCV' 19]** |[`[pdf]`](http://openaccess.thecvf.com/content_ICCV_2019/papers/Tan_Learning_to_Rank_Proposals_for_Object_Detection_ICCV_2019_paper.pdf)

- **[WSOD2]** WSOD2: Learning Bottom-Up and Top-Down Objectness Distillation for Weakly-Supervised Object Detection	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1909.04972.pdf)

- **[ClusDet]** Clustered Object Detection in Aerial Images	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.08008.pdf)

- Towards Precise End-to-End Weakly Supervised Object Detection Network	| **[ICCV' 19]** |[`[pdf]`](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_Towards_Precise_End-to-End_Weakly_Supervised_Object_Detection_Network_ICCV_2019_paper.pdf)

- Few-Shot Object Detection via Feature Reweighting	 | **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1812.01866.pdf)

- **[Objects365]** Objects365: A Large-Scale, High-Quality Dataset for Object Detection	| **[ICCV' 19]** |[`[pdf]`](http://openaccess.thecvf.com/content_ICCV_2019/papers/Shao_Objects365_A_Large-Scale_High-Quality_Dataset_for_Object_Detection_ICCV_2019_paper.pdf)

- **[EGNet]** EGNet: Edge Guidance Network for Salient Object Detection	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1908.08297.pdf)

- Optimizing the F-Measure for Threshold-Free Salient Object Detection	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1805.07567.pdf)

- Sequence Level Semantics Aggregation for Video Object Detection	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1907.06390.pdf)

- **[NOTE-RCNN]** NOTE-RCNN: NOise Tolerant Ensemble RCNN for Semi-Supervised Object Detection | **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1812.00124.pdf)	

- Enriched Feature Guided Refinement Network for Object Detection	| **[ICCV' 19]** |[`[pdf]`](http://openaccess.thecvf.com/content_ICCV_2019/papers/Nie_Enriched_Feature_Guided_Refinement_Network_for_Object_Detection_ICCV_2019_paper.pdf)

- **[POD]** POD: Practical Object Detection With Scale-Sensitive Network	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1909.02225.pdf)	

- **[FCOS]** FCOS: Fully Convolutional One-Stage Object Detection	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.01355.pdf)	

- **[RepPoints]** RepPoints: Point Set Representation for Object Detection	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.11490.pdf)	

- Better to Follow, Follow to Be Better: Towards Precise Supervision of Feature Super-Resolution for Small Object Detection	| **[ICCV' 19]** |[`[pdf]`](http://openaccess.thecvf.com/content_ICCV_2019/papers/Noh_Better_to_Follow_Follow_to_Be_Better_Towards_Precise_Supervision_ICCV_2019_paper.pdf)

- Weakly Supervised Object Detection With Segmentation Collaboration	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.00551.pdf)	

- Leveraging Long-Range Temporal Relationships Between Proposals for Video Object Detection	| **[ICCV' 19]** |[`[pdf]`](http://openaccess.thecvf.com/content_ICCV_2019/papers/Shvets_Leveraging_Long-Range_Temporal_Relationships_Between_Proposals_for_Video_Object_Detection_ICCV_2019_paper.pdf)

- Detecting 11K Classes: Large Scale Object Detection Without Fine-Grained Bounding Boxes	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1908.05217.pdf)	

- **[C-MIDN]** C-MIDN: Coupled Multiple Instance Detection Network With Segmentation Guidance for Weakly Supervised Object Detection	| **[ICCV' 19]** |[`[pdf]`]()

- Meta-Learning to Detect Rare Objects	| **[ICCV' 19]** |[`[pdf]`](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Meta-Learning_to_Detect_Rare_Objects_ICCV_2019_paper.pdf)

- **[Cap2Det]** Cap2Det: Learning to Amplify Weak Caption Supervision for Object Detection | **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1907.10164v1.pdf)

- **[Gaussian YOLOv3]** Gaussian YOLOv3: An Accurate and Fast Object Detector using Localization Uncertainty for Autonomous Driving | **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.04620.pdf) [`[official code - c]`](https://github.com/jwchoi384/Gaussian_YOLOv3)

- **[FreeAnchor]** FreeAnchor: Learning to Match Anchors for Visual Object Detection | **[NeurIPS' 19]** |[`[pdf]`](https://arxiv.org/pdf/1909.02466v1.pdf)

- Memory-oriented Decoder for Light Field Salient Object Detection | **[NeurIPS' 19]** |[`[pdf]`](http://papers.nips.cc/paper/8376-memory-oriented-decoder-for-light-field-salient-object-detection.pdf)

- One-Shot Object Detection with Co-Attention and Co-Excitation | **[NeurIPS' 19]** |[`[pdf]`](http://papers.nips.cc/paper/8540-one-shot-object-detection-with-co-attention-and-co-excitation.pdf)

- **[DetNAS]** DetNAS: Backbone Search for Object Detection | **[NeurIPS' 19]** |[`[pdf]`](https://arxiv.org/pdf/1903.10979v4.pdf)

- Consistency-based Semi-supervised Learning for Object detection | **[NeurIPS' 19]** |[`[pdf]`](https://papers.nips.cc/paper/9259-consistency-based-semi-supervised-learning-for-object-detection.pdf)

- **[NATS]** Efficient Neural Architecture Transformation Searchin Channel-Level for Object Detection | **[NeurIPS' 19]** |[`[pdf]`](https://arxiv.org/pdf/1909.02293.pdf)

- **[AA]** Learning Data Augmentation Strategies for Object Detection | **[arXiv' 19]** |[`[pdf]`](https://arxiv.org/pdf/1906.11172.pdf)

- **[Spinenet]** Spinenet: Learning scale-permuted backbone for recognition and localization | **[arXiv' 19]** |[`[pdf]`](https://arxiv.org/pdf/1912.05027.pdf)

- Object Detection in 20 Years: A Survey | **[arXiv' 19]** |[`[pdf]`](https://arxiv.org/pdf/1905.05055.pdf)

## 2020
- **[Spiking-YOLO]** Spiking-YOLO: Spiking Neural Network for Real-time Object Detection | **[AAAI' 20]** |[`[pdf]`](https://arxiv.org/pdf/1903.06530.pdf)

- Tell Me What They're Holding: Weakly-supervised Object Detection with Transferable Knowledge from Human-object Interaction | **[AAAI' 20]** |[`[pdf]`](https://arxiv.org/pdf/1911.08141v1.pdf)

- **[CBnet]** Cbnet: A novel composite backbone network architecture for object detection | **[AAAI' 20]** |[`[pdf]`](https://arxiv.org/pdf/1909.03625.pdf)

- **[Distance-IoU Loss]** Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression | **[AAAI' 20]** |[`[pdf]`](https://arxiv.org/pdf/1911.08287v1.pdf)

- Computation Reallocation for Object Detection | **[ICLR' 20]** |[`[pdf]`](https://openreview.net/pdf?id=SkxLFaNKwB)

- **[YOLOv4]** YOLOv4: Optimal Speed and Accuracy of Object Detection | **[arXiv' 20]** |[`[pdf]`](https://arxiv.org/pdf/2004.10934.pdf)

- Few-Shot Object Detection With Attention-RPN and Multi-Relation Detector | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/1908.01998.pdf)

- Large-Scale Object Detection in the Wild From Imbalanced Multi-Labels | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/2005.08455.pdf)

- Bridging the Gap Between Anchor-Based and Anchor-Free Detection via Adaptive Training Sample Selection | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/1912.02424.pdf)

- Rethinking Classification and Localization for Object Detection	 | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/1904.06493.pdf)

- Multiple Anchor Learning for Visual Object Detection | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/1912.02252.pdf)

- **[CentripetalNet]** CentripetalNet: Pursuing High-Quality Keypoint Pairs for Object Detection	 | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/2003.09119.pdf)

- Learning From Noisy Anchors for One-Stage Object Detection | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/1912.05086.pdf)

- **[EfficientDet]** EfficientDet: Scalable and Efficient Object Detection | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/1911.09070.pdf)

- Overcoming Classifier Imbalance for Long-Tail Object Detection With Balanced Group Softmax | **[CVPR' 20]** 

- Dynamic Refinement Network for Oriented and Densely Packed Object Detection | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/2005.09973.pdf)

- Noise-Aware Fully Webly Supervised Object Detection	 | **[CVPR' 20]** 

- **[Hit-Detector]** Hit-Detector: Hierarchical Trinity Architecture Search for Object Detection	 | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/2003.11818.pdf)

- **[D2Det]** D2Det: Towards High Quality Object Detection and Instance Segmentation | **[CVPR' 20]** 

- Prime Sample Attention in Object Detection | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/1904.04821.pdf)

- Don’t Even Look Once: Synthesizing Features for Zero-Shot Detection	 | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/1911.07933.pdf)

- Exploring Categorical Regularization for Domain Adaptive Object Detection	 | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/2003.09152.pdf)

- **[SP-NAS]** SP-NAS: Serial-to-Parallel Backbone Search for Object Detection	 | **[CVPR' 20]** 

- **[NAS-FCOS]** NAS-FCOS: Fast Neural Architecture Search for Object Detection | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/1906.04423.pdf)

- **[DR Loss]** DR Loss: Improving Object Detection by Distributional Ranking	 | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/1907.10156.pdf)

- Detection in Crowded Scenes: One Proposal, Multiple Predictions	 | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/2003.09163.pdf)

- **[AugFPN]** AugFPN: Improving Multi-Scale Feature Learning for Object Detection	 | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/1912.05384.pdf)

- Robust Object Detection Under Occlusion With Context-Aware CompositionalNets	 | **[CVPR' 20]** 

- Cross-Domain Document Object Detection: Benchmark Suite and Method | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/2003.13197.pdf)

- Exploring Bottom-Up and Top-Down Cues With Attentive Learning for Webly Supervised Object Detection	 | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/2003.09790.pdf)

- **[SLV]** SLV: Spatial Likelihood Voting for Weakly Supervised Object Detection	 | **[CVPR' 20]** 

- **[HAMBox]** HAMBox: Delving Into Mining High-Quality Anchors on Face Detection	 | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/1912.09231.pdf)

- **[Context R-CNN]** Context R-CNN: Long Term Temporal Context for Per-Camera Object Detection	 | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/1912.03538.pdf)

- Mixture Dense Regression for Object Detection and Human Pose Estimation	 | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/1912.00821.pdf)

- Offset Bin Classification Network for Accurate Object Detection	 | **[CVPR' 20]** 

- **[NETNet]** NETNet: Neighbor Erasing and Transferring Network for Better Single Shot Object Detection	 | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/2001.06690.pdf)

- Scale-Equalizing Pyramid Convolution for Object Detection	 | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/2005.03101.pdf)

- Temporal-Context Enhanced Detection of Heavily Occluded Pedestrians	 | **[CVPR' 20]** |[`[pdf]`](https://cse.buffalo.edu/~jsyuan/papers/2020/TFAN.pdf)

- **[MnasFPN]** MnasFPN: Learning Latency-Aware Pyramid Architecture for Object Detection on Mobile Devices	 | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/1912.01106.pdf)

- Physically Realizable Adversarial Examples for LiDAR Object Detection	 | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/2004.00543.pdf)

- Cross-domain Object Detection through Coarse-to-Fine Feature Adaptation	 | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/2003.10275.pdf)

- Incremental Few-Shot Object Detection	 | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/2003.04668.pdf)

- Where, What, Whether: Multi-Modal Learning Meets Pedestrian Detection	 | **[CVPR' 20]** 

- Cylindrical Convolutional Networks for Joint Object Detection and Viewpoint Estimation	 | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/2003.11303.pdf)

- Learning a Unified Sample Weighting Network for Object Detection	 | **[CVPR' 20]** 

- Seeing without Looking: Contextual Rescoring of Object Detections for AP Maximization	 | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/1912.12290.pdf)

- DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution	 | **[arXiv' 20]** |[`[pdf]`](https://arxiv.org/pdf/2006.02334v1.pdf)

- **[DETR]** End-to-End Object Detection with Transformers	| **[ECCV' 20]** |[`[pdf]`](https://arxiv.org/pdf/2005.12872.pdf)

- Suppress and Balance: A Simple Gated Network for Salient Object Detection	| **[ECCV' 20]** |[`[code]`](https://github.com/Xiaoqi-Zhao-DLUT/GateNet-RGB-Saliency)

- **[BorderDet]** BorderDet: Border Feature for Dense Object Detection | **[ECCV' 20]** 
- Corner Proposal Network for Anchor-free, Two-stage Object Detection	| **[ECCV' 20]** 
- A General Toolbox for Understanding Errors in Object Detection	| **[ECCV' 20]** 
- **[Chained-Tracker]** Chained-Tracker: Chaining Paired Attentive Regression Results for End-to-End Joint Multiple-Object Detection and Tracking		| **[ECCV' 20]** 
- Side-Aware Boundary Localization for More Precise Object Detection	| **[ECCV' 20]** 
- **[PIoU]** PIoU Loss: Towards Accurate Oriented Object Detection in Complex Environments	| **[ECCV' 20]** 
- **[AABO]** AABO: Adaptive Anchor Box Optimization for Object Detection via Bayesian Sub-sampling	| **[ECCV' 20]** 
- Highly Efficient Salient Object Detection with 100K Parameters		| **[ECCV' 20]** 
- **[GeoGraph]** GeoGraph: Learning graph-based multi-view object detection with geometric cues end-to-end		| **[ECCV' 20]** 
- Many-shot from Low-shot: Learning to Annotate using Mixed Supervision for Object Detection		| **[ECCV' 20]** 
- Cheaper Pre-training Lunch: An Efficient Paradigm for Object Detection	| **[ECCV' 20]** 
- Arbitrary-Oriented Object Detection with Circular Smooth Label	| **[ECCV' 20]** 
- Soft Anchor-Point Object Detection		| **[ECCV' 20]** 
- Object Detection with a Unified Label Space from Multiple Datasets	| **[ECCV' 20]** 
- **[MimicDet]** MimicDet: Bridging the Gap Between One-Stage and Two-Stage Object Detection		| **[ECCV' 20]** 
- Prior-based Domain Adaptive Object Detection for Hazy and Rainy Conditions		| **[ECCV' 20]** 
- **[Dynamic R-CNN]** Dynamic R-CNN: Towards High Quality Object Detection via Dynamic Training		| **[ECCV' 20]** 
- **[OS2D]** OS2D: One-Stage One-Shot Object Detection by Matching Anchor Features		| **[ECCV' 20]** 
- Multi-Scale Positive Sample Refinement for Few-Shot Object Detection		| **[ECCV' 20]** 
- Few-Shot Object Detection and Viewpoint Estimation for Objects in the Wild		| **[ECCV' 20]** 
- Collaborative Training between Region Proposal Localization and Classification for Domain Adaptive Object Detection		| **[ECCV' 20]** 
- Two-Stream Active Query Suggestion for Large-Scale Object Detection in Connectomics		| **[ECCV' 20]** 
- **[FDTS]** FDTS: Fast Diverse-Transformation Search for Object Detection and Beyond		| **[ECCV' 20]** 
- Dual refinement underwater object detection network		| **[ECCV' 20]** 
- **[APRICOT]** APRICOT: A Dataset of Physical Adversarial Attacks on Object Detection		| **[ECCV' 20]** 
- Large Batch Optimization for Object Detection: Training COCO in 12 Minutes		| **[ECCV' 20]** 
- Hierarchical Context Embedding for Region-based Object Detection		| **[ECCV' 20]** 
- Pillar-based Object Detection for Autonomous Driving		| **[ECCV' 20]** 
- Dive Deeper Into Box for Object Detection		| **[ECCV' 20]** 
- Domain Adaptive Object Detection via Asymmetric Tri-way Faster-RCNN		| **[ECCV' 20]** 
- Probabilistic Anchor Assignment with IoU Prediction for Object Detection		| **[ECCV' 20]** 
- **[HoughNet]** HoughNet: Integrating near and long-range evidence for bottom-up object detection		| **[ECCV' 20]** 
- **[LabelEnc]** LabelEnc: A New Intermediate Supervision Method for Object Detection		| **[ECCV' 20]** 
- Boosting Weakly Supervised Object Detection with Progressive Knowledge Transfer		| **[ECCV' 20]** 
- On the Importance of Data Augmentation for Object Detection		| **[ECCV' 20]** 
- Adaptive Object Detection with Dual Multi-Label Prediction		| **[ECCV' 20]** 
- Quantum-soft QUBO Suppression for Accurate Object Detection		| **[ECCV' 20]** 
- Improving Object Detection with Selective Self-supervised Self-training		| **[ECCV' 20]** 


##

## Dataset Papers
Statistics of commonly used object detection datasets. The Table came from [this survey paper](https://arxiv.org/pdf/1809.02165v1.pdf).

<table>
<thead>
  <tr>
    <th rowspan=2>Challenge</th>
    <th rowspan=2 width=80>Object Classes</th>
    <th colspan=3>Number of Images</th>
    <th colspan=2>Number of Annotated Images</th>
  </tr>
  <tr>
    <th>Train</th>
    <th>Val</th>
    <th>Test</th>
    <th>Train</th>
    <th>Val</th>
  </tr>
</thead>
<tbody>

<!-- PASCAL VOC Object Detection Challenge -->
<tr><th colspan=7>PASCAL VOC Object Detection Challenge</th></tr>
<tr><td> VOC07 </td><td> 20 </td><td> 2,501 </td><td> 2,510 </td><td>  4,952 </td><td>   6,301 (7,844) </td><td>   6,307 (7,818) </td></tr>
<tr><td> VOC08 </td><td> 20 </td><td> 2,111 </td><td> 2,221 </td><td>  4,133 </td><td>   5,082 (6,337) </td><td>   5,281 (6,347) </td></tr>
<tr><td> VOC09 </td><td> 20 </td><td> 3,473 </td><td> 3,581 </td><td>  6,650 </td><td>   8,505 (9,760) </td><td>   8,713 (9,779) </td></tr>
<tr><td> VOC10 </td><td> 20 </td><td> 4,998 </td><td> 5,105 </td><td>  9,637 </td><td> 11,577 (13,339) </td><td> 11,797 (13,352) </td></tr>
<tr><td> VOC11 </td><td> 20 </td><td> 5,717 </td><td> 5,823 </td><td> 10,994 </td><td> 13,609 (15,774) </td><td> 13,841 (15,787) </td></tr>
<tr><td> VOC12 </td><td> 20 </td><td> 5,717 </td><td> 5,823 </td><td> 10,991 </td><td> 13,609 (15,774) </td><td> 13,841 (15,787) </td></tr>

<!-- ILSVRC Object Detection Challenge -->
<tr><th colspan=7>ILSVRC Object Detection Challenge</th></tr>
<tr><td> ILSVRC13 </td><td> 200 </td><td> 395,909 </td><td> 20,121 </td><td> 40,152 </td><td> 345,854 </td><td> 55,502 </td></tr>
<tr><td> ILSVRC14 </td><td> 200 </td><td> 456,567 </td><td> 20,121 </td><td> 40,152 </td><td> 478,807 </td><td> 55,502 </td></tr>
<tr><td> ILSVRC15 </td><td> 200 </td><td> 456,567 </td><td> 20,121 </td><td> 51,294 </td><td> 478,807 </td><td> 55,502 </td></tr>
<tr><td> ILSVRC16 </td><td> 200 </td><td> 456,567 </td><td> 20,121 </td><td> 60,000 </td><td> 478,807 </td><td> 55,502 </td></tr>
<tr><td> ILSVRC17 </td><td> 200 </td><td> 456,567 </td><td> 20,121 </td><td> 65,500 </td><td> 478,807 </td><td> 55,502 </td></tr>

<!-- MS COCO Object Detection Challenge -->
<tr><th colspan=7>MS COCO Object Detection Challenge</th></tr>
<tr><td> MS COCO15 </td><td> 80 </td><td>  82,783 </td><td> 40,504 </td><td> 81,434 </td><td> 604,907 </td><td> 291,875 </td></tr>
<tr><td> MS COCO16 </td><td> 80 </td><td>  82,783 </td><td> 40,504 </td><td> 81,434 </td><td> 604,907 </td><td> 291,875 </td></tr>
<tr><td> MS COCO17 </td><td> 80 </td><td> 118,287 </td><td>  5,000 </td><td> 40,670 </td><td> 860,001 </td><td>  36,781 </td></tr>
<tr><td> MS COCO18 </td><td> 80 </td><td> 118,287 </td><td>  5,000 </td><td> 40,670 </td><td> 860,001 </td><td>  36,781 </td></tr>

<!-- Open Images Object Detection Challenge -->
<tr><th colspan=7>Open Images Object Detection Challenge</th></tr>
<tr><td> OID18 </td><td> 500 </td><td> 1,743,042 </td><td> 41,620 </td><td> 125,436 </td><td> 12,195,144 </td><td> ― </td></tr>

  </tbody>
</table>

The papers related to datasets used mainly in Object Detection are as follows.

- **[PASCAL VOC]** The PASCAL Visual Object Classes (VOC) Challenge | **[IJCV' 10]** | [`[pdf]`](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf)

- **[PASCAL VOC]** The PASCAL Visual Object Classes Challenge: A Retrospective | **[IJCV' 15]** | [`[pdf]`](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham15.pdf) | [`[link]`](http://host.robots.ox.ac.uk/pascal/VOC/)

- **[ImageNet]** ImageNet: A Large-Scale Hierarchical Image Database| **[CVPR' 09]** | [`[pdf]`](http://www.image-net.org/papers/imagenet_cvpr09.pdf)

- **[ImageNet]** ImageNet Large Scale Visual Recognition Challenge | **[IJCV' 15]** | [`[pdf]`](https://arxiv.org/pdf/1409.0575.pdf) | [`[link]`](http://www.image-net.org/challenges/LSVRC/)

- **[COCO]** Microsoft COCO: Common Objects in Context | **[ECCV' 14]** | [`[pdf]`](https://arxiv.org/pdf/1405.0312.pdf) | [`[link]`](http://cocodataset.org/)

- **[Open Images]** The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale | **[arXiv' 18]** | [`[pdf]`](https://arxiv.org/pdf/1811.00982v1.pdf) | [`[link]`](https://storage.googleapis.com/openimages/web/index.html)

- **[DOTA]** DOTA: A Large-scale Dataset for Object Detection in Aerial Images | **[CVPR' 18]** | [`[pdf]`](https://arxiv.org/pdf/1711.10398v3.pdf) | [`[link]`](https://captain-whu.github.io/DOTA/)

- **[Objects365]** Objects365: A Large-Scale, High-Quality Dataset for Object Detection	| **[ICCV' 19]** | [`[link]`](https://www.biendata.com/competition/objects365/)

##

## Contact & Feedback

If you have any suggestions about papers, feel free to mail me :)

- [e-mail](mailto:Hoseong.Lee@cognex.com)
- [blog](https://hoya012.github.io/)
- [pull request](https://github.com/hoya012/deep_learning_object_detection/pulls)
