
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import os
import pathlib
import glob
import fnmatch
# import io
# import argparse
# import scipy.misc
import numpy as np
# from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
import time
import cv2
import warnings
from flask import Flask, render_template, Response, request
warnings.filterwarnings('ignore')

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

print(tf.__version__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)

tf.get_logger().setLevel('ERROR')         # Suppress TensorFlow logging (2)

app=Flask(__name__)

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.
  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.
  Args:
    path: the file path to the image
  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """

  return np.array(Image.open(path))


# @title Choose the model to use, then evaluate the cell.
MODEL_NAME  = 'model'

DATA_DIR = os.getcwd()
MODELS_DIR = os.path.join(DATA_DIR, 'workspace/exported-models')
PATH_TO_CKPT = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'checkpoint/'))
PATH_TO_CFG = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'pipeline.config'))
# PATH_TO_SAVED_MODEL = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'saved_model'))
print('Loading model... ', end='')
start_time = time.time()

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

def get_model_detection_function(model):
  """Get a tf.function for detection."""

  @tf.function
  def detect_fn(image):
      """Detect objects in image."""

      image, shapes = detection_model.preprocess(image)
      prediction_dict = detection_model.predict(image, shapes)
      detections = detection_model.postprocess(prediction_dict, shapes)

      return detections, prediction_dict, tf.reshape(shapes, [-1])
  return detect_fn

# MODEL_NAME = 'workspace/annotations'
# LABEL_FILENAME = 'label_map.pbtxt'
# PATH_TO_LABELS = os.path.join(MODELS_DIRx, os.path.join(MODEL_NAME, LABEL_FILENAME))
PATH_TO_LABELS = "workspace/annotations/label_map.pbtxt"
# PATH_TO_LABELS = "/home/nwoke/Documents/git_cloned/Github/models/research/object_detection/data/mscoco_label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
detect_fn = get_model_detection_function(detection_model)
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))


cap = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('stb_out.avi',fourcc, 20.0, (640,480))

def get_detections():
  while True:
    
    # Read frame from camera
    re, image_np = cap.read()
    if not re:
      break
    else:
    # image_np = load_image_into_numpy_array(image_np)
      image_np_expanded = np.expand_dims(image_np, axis=0)

  #  Expand dimensions since the model expects images to have shape: [1, None, None, 3]    
      input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    # input_tensor = tf.convert_to_tensor(image_np)

    # input_tensor = input_tensor[tf.newaxis, ...]

    

  # # Things to try:
  # # Flip horizontally
  # image_np = np.fliplr(image_np).copy()

  # Convert image to grayscale
  # image_np = np.tile(
  #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

      detections, predictions_dict, shapes = detect_fn(input_tensor)

      

      label_id_offset = 1
      image_np_with_detections = image_np.copy()

      viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections,
      detections['detection_boxes'][0].numpy(),
      (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
      detections['detection_scores'][0].numpy(),
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=.30,
      agnostic_mode=False)


      ret, buffer = cv2.imencode('.jpg', image_np_with_detections)
      frame = buffer.tobytes() 
    
    yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
      
  #   scores = detections['detection_scores'][0]
  #   scores = scores[scores> 0.5]
  #   print(scores)
  #   # print(detections['detection_boxes'][0][above])

  #   # Display the resulting frame
  #     # out.write(image_np_with_detections)
  #   cv2.imshow('frame',image_np_with_detections)
  #   if cv2.waitKey(1) & 0xFF == ord('q'):
  #     break

  # cap.release()
  # cv2.destroyAllWindows() 


@app.route('/')
def index():
  return render_template('index.html')

@app.route('/detections')
def video():
  return Response(get_detections(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=='__main__':
  # get_detections()
  app.run(host='0.0.0.0', port=2204, threaded=True)