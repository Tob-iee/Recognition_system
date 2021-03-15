
# %matplotlib inline
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import os
import pathlib
import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
import time
import cv2

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


print(tf.__version__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)

tf.get_logger().setLevel('ERROR')         # Suppress TensorFlow logging (2)

def download_images(image_dir):
#     filenames = ['image1.jpg', 'image2.jpg']
    image_paths = []
    for filename in pathlib.Path(image_dir).glob('*.jpg'):
#         if filename.endswith(".jpg"):
#             image_path = pathlib.Path(filename)
        image_paths.append(str(filename))
    return image_paths

image_dir = '/home/nwoke/Documents/git_cloned/Github/face_recognizer/new_images'
IMAGE_PATHS = download_images(image_dir)
# print(IMAGE_PATHS)

# # Enable GPU dynamic memory allocation
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# @title Choose the model to use, then evaluate the cell.
MODELS = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
# model_display_name = 'Mobilenet_with_keypoints' # @param ['centernet_with_keypoints', 'centernet_without_keypoints']
MODEL_NAME = MODELS

# pipeline_config = os.path.join('/home/nwoke/Documents/git_cloned/Github/models/research/object_detection/configs/tf2/',
#                                 model_name + '.config')
DATA_DIR = os.getcwd()
# print(DATA_DIR)
MODELS_DIR = os.path.join(DATA_DIR, 'model')
# print(MODELS_DIR)
PATH_TO_CKPT = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'checkpoint/'))
PATH_TO_CFG = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'pipeline.config'))
print('Loading model... ', end='')
start_time = time.time()

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])
    # return detections

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))


# PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))
PATH_TO_LABELS = "/home/nwoke/Documents/git_cloned/Github/models/research/object_detection/data/mscoco_label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

cap = cv2.VideoCapture(0)


while True:
    # Read frame from camera
    ret, image_np = cap.read()

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
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

    # Display output
    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()