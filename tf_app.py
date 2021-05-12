
# %matplotlib inline
import tensorflow as tf
import matplotlib
# matplotlib.use('TkAgg')
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
import warnings
warnings.filterwarnings('ignore')

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


print(tf.__version__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)

tf.get_logger().setLevel('ERROR')         # Suppress TensorFlow logging (2)

# def download_images(image_dir):
# #     filenames = ['image1.jpg', 'image2.jpg']
#     image_paths = []
#     for filename in pathlib.Path(image_dir).glob('*.jpg'):
# #         if filename.endswith(".jpg"):
# #             image_path = pathlib.Path(filename)
#         image_paths.append(str(filename))
#     return image_paths

def load_image_into_numpy_array(image):
  """Load an image from file into a numpy array.
  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.
  Args:
    path: the file path to the image
  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  #img_data = tf.io.gfile.GFile(path, 'rb').read()
  #image = Image.open(BytesIO(img_data))
  (im_width, im_height, channel) = image.shape
  return image.astype(np.uint8)

# def run_inference_for_single_image(model, image):
#   image = np.asarray(image)
#   # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
#   input_tensor = tf.convert_to_tensor(image)
#   # The model expects a batch of images, so add an axis with `tf.newaxis`.
#   input_tensor = input_tensor[tf.newaxis,...]
 
#   # Run inference
#   model_fn = model.signatures['serving_default']
#   output_dict = model_fn(input_tensor)
 
#   # All outputs are batches tensors.
#   # Convert to numpy arrays, and take index [0] to remove the batch dimension.
#   # We're only interested in the first num_detections.
#   num_detections = int(output_dict.pop('num_detections'))
#   output_dict = {key:value[0, :num_detections].numpy() 
#                  for key,value in output_dict.items()}
#   output_dict['num_detections'] = num_detections
 
#   # detection_classes should be ints.
#   output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
#   # Handle models with masks:
#   if 'detection_masks' in output_dict:
#     # Reframe the the bbox mask to the image size.
#     detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
#               output_dict['detection_masks'], output_dict['detection_boxes'],
#                image.shape[0], image.shape[1])      
#     detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
#                                        tf.uint8)
#     output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
     
#   return output_dict

# image_dir = '/home/nwoke/Documents/git_cloned/Github/face_recognizer/new_images'
# IMAGE_PATHS = download_images(image_dir)
# print(IMAGE_PATHS)

# # Enable GPU dynamic memory allocation
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# @title Choose the model to use, then evaluate the cell.
# MODELS = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
# model_display_name = 'Mobilenet_with_keypoints' # @param ['centernet_with_keypoints', 'centernet_without_keypoints']
MODEL_NAME = 'my_model'

# pipeline_config = os.path.join('/home/nwoke/Documents/git_cloned/Github/models/research/object_detection/configs/tf2/',
#                                 model_name + '.config')
DATA_DIR = os.getcwd()
MODELS_DIR = os.path.join(DATA_DIR, 'Model')
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

detect_fn = get_model_detection_function(detection_model)
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

LABEL_FILENAME = "label_map.pbtxt"
PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))
#PATH_TO_LABELS = "C:/Users\anjuw\Documents\programming\Github\Recognition_system\Model\my_model\label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('stb_out.avi',fourcc, 20.0, (640,480))

while True:
    for _ in range(10):
        # Read frame from camera
        ret, image_np = cap.read()
        image_np = load_image_into_numpy_array(image_np)

    #  Expand dimensions since the model expects images to have shape: [1, None, None, 3]    
        input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32)
    # image_np_expanded = np.expand_dims(image_np, axis=0)

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

  # Display the resulting frame
    out.write(image_np_with_detections)
    cv2.imshow('frame',image_np_with_detections)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # cv2.imshow('object detection', image_np)

    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()