## Efficient Lane and Path Detection Learning Model and Architecture for Autonomous Vehicle 
=====================================

## The Model

This research is done based on supercombo model commit [83112b47-3b23-48e4-b65b-8c058766f4c1/100](https://github.com/commaai/openpilot/commit/a420d4bcbb299f69605de4628657261706a89e6b) by comma.ai

### Model Summary

<div align="center">
      <a href="">
     <img 
      src="https://github.com/iXcess/driving_model/blob/master/assets/model_summary1.jpg" 
      alt="Model Summary" 
      style="width:100%;">
      </a>
      <a href="">
     <img 
      src="https://github.com/iXcess/driving_model/blob/master/assets/model_summary2.jpg" 
      alt="Model Summary" 
      style="width:100%;">
      </a>
      <a href="">
     <img 
      src="https://github.com/iXcess/driving_model/blob/master/assets/model_summary3.jpg" 
      alt="Model Summary" 
      style="width:100%;">
      </a>
      <a href="">
     <img 
      src="https://github.com/iXcess/driving_model/blob/master/assets/model_summary4.jpg" 
      alt="Model Summary" 
      style="width:100%;">
      </a>
</div>

#### Inputs
The model has 4 inputs.
1. Size (12,128,256) image
2. One hot coding input of length 8
3. rnn_state or the vehicle state from openpilot of length 512
4. traffic_convention of length 2

The model image input will first be converted to YUV_I420 from BGR, this will reduce the dimension of the image by 1.
The image is then transformed from eon_intrinsic frame to medmodel_intrinsic frame, a copy is saved for the recurrent network.
The output of the transformed image is stacked together to form a (1,12,128,256) tensor.


#### Outputs
The output of the model has 11 outputs
1. 192 outputs points spaced 1m apart from the car reference frame of the path
2. 192 left lane points spaced 1m apart from the car reference frame
3. 192 right lane points spaced 1m apart from the car reference frame
4. lead car state vector of the lead car of length 58
5. longitudinal_x of length 200
6. longitudinal_v of length 200
7. longitudinal_a of length 200
8. meta of length 4
9. snpe_pleaser2 of length 4
10. pose of length 32, used for posenet to get the homogenous transformation matrix between two device frame (attached to the camera) and the vehicle frame, read about [calibration](https://github.com/commaai/openpilot/tree/master/common/transformations)
11. add3 (unknown what is this currently)

### Model Graph Visualisation

<div align="center">
      <a href="">
     <img 
      src="https://github.com/iXcess/driving_model/blob/master/assets/graph_out1.jpg" 
      alt="Model Graph" 
      style="width:100%;margin-top: 100px;">
      </a>
      <a href="">
     <img 
      src="https://github.com/iXcess/driving_model/blob/master/assets/graph_out2.jpg" 
      alt="Model Graph" 
      style="width:80%;margin-top: 100px;">
      </a>
</div>

## Objective 1: To Design a Universal Learning Model for Autonomous Vehicle Applications.

Under this objective, the designed model must be able to produce high accuracy in lane feature extraction. The model must then be able to do transfer learning to other critical tasks such as lead vehicle state vector estimation and driver's state monitoring. The model must include a recurrent unit.

#### EfficientNet
The comparison of EfficientNet and other state of the other models are shown below:

<div align="center">
      <a href="">
     <img 
      src="https://github.com/iXcess/driving_model/blob/master/assets/efficientnet_graph.jpg" 
      alt="EfficientNet graph" 
      style="width:60%;margin: 50px;">
      </a>
</div>

In November 2019, Andrej Karpathy, the Senior Director of Artificial Intelligence at Tesla mentioned in the [Pytorch Devcon](https://youtu.be/oBklltKXtDE?t=170) that the autopilot has most of its models based off a ResNet50. Openpilot by comma.ai has been using ResNet18 for quite awhile until recently they have switched to the use of EfficientNet-B2.

EfficientNets are family of models which was optimised accuracy and floating operation per seconds (FLOPs). The baseline model was developed by leveraging a multi-objective neural architecture search. This has became the baseline called the EfficientNet-B0. [The paper](https://arxiv.org/pdf/1905.11946.pdf) has demonstrated that the EfficientNet-B7 achieved state-of-the-art 84.3% top-1 accuracy on ImageNet, while being 8.4x smaller
and 6.1x faster on inference than best existing ConvNet. Not only that, this baseline model transfer well and achieve state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters. For this reason, the EfficientNet is chosen for its efficiency and its application universality with transfer learning.

The decision between EfficientNet-B2 to B4 is still being considered considering the hardware constraints.

## Installation Guide

It is recommended to use Python 3.6 or above.

1. Install all the necessary dependencies.

```
pip3 install -r requirements.txt
```
2. Run the program

``` 
python3 dynamic_lane.py <path-to-sample-hevc> 
```

## To-do

- Update the requirements.txt
- Train a simple model which can do path prediction, driver monitoring and lead car state vector estimation.
- path_predict_simple_efficientnet. Input : 


## Related Research

[Learning a driving simulator](https://arxiv.org/abs/1608.01230)
[EfficientNet](https://arxiv.org/pdf/1905.11946.pdf)

## Credits

[comma.ai for supercombo model](https://github.com/commaai/openpilot/blob/master/models/supercombo.keras)
