# Recognition_system

This application using the local camera system to take video input to detects human faces and streams it through your local host on your browse by using the flasl API


## Guide for running the app
### Requirements
flask\
opencv-contrib-python-headless\
matplotlib

### Detection Model used
The models that is used is tensorflow model that has been re-trained model, And it was gotten from the list of available pre-trained models on the Tensorflow object detection API repository ([Link](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)). For tutorials on how to re-train a tensorflow model this could serve as a good guide ([Link](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html))

### To run the app
```
Python3 scripts/tf_app.py
```
### To see output
paste this url in your browser
```
http://0.0.0.0:2204/detections
```

## Observation
When closer to the camera the detection accuracy is higher and there are lags in the detection between frame intervals.
