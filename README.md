Efficient Lane and Path Detection Learning Model and Architecture for Autonomous Vehicle 
=====================================
## Demo
<div align="center">
      <a href="">
     <img 
      src="https://github.com/iXcess/driving_model/blob/master/assets/demo.jpg" 
      alt="Demo" 
      style="width:100%;">
      </a>
</div>

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

The model has 4 inputs.
1. Size (12,128,256) image
2. One hot coding input of length 8
3. rnn_state or the vehicle state from openpilot of length 512
4. traffic_convention of length 2

The output of the model has 10 outputs
1. 192 outputs points spaced 1m apart from the car reference frame of the path
2. 192 left lane points spaced 1m apart from the car reference frame
3. 192 right lane points spaced 1m apart from the car reference frame
4. lead car state vector of the lead car of length 58
5. longitudinal_x of length 200
6. longitudinal_v of length 200
7. longitudinal_a of length 200
8. meta of length 4
9. snpe_pleaser2 of length 4
10. pose of length 32, used for posenet to get the homogenous transformation matrix between two input frames

### Model Graph Visualisation

<div align="center">
      <a href="">
     <img 
      src="https://github.com/iXcess/driving_model/blob/master/assets/graph_out1.jpg" 
      alt="Model Graph" 
      style="width:100%;">
      </a>
      <a href="">
     <img 
      src="https://github.com/iXcess/driving_model/blob/master/assets/graph_out2.jpg" 
      alt="Model Graph" 
      style="width:100%;">
      </a>
</div>

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
- Write about the findings


## Related Research

[Learning a driving simulator](https://arxiv.org/abs/1608.01230)

## Credits

[comma.ai for supercombo model](https://github.com/commaai/openpilot/blob/master/models/supercombo.keras)
