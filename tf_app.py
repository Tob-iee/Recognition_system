# import the neccessary packages
import os
import time
import cv2
import numpy as np
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils

print("The tensorflow version is "+ tf.__version__)
#Complete the tensorflow setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)

tf.get_logger().setLevel('ERROR')         # Suppress TensorFlow logging (2)

# Get the Current directory 
Current_directory = os.getcwd()

print("The current Directory is "+Current_directory)

filenames = map(lambda x: '\\new_images\\' + x, 
                ('image 1.jpg', 'image 2.jpg',
                'image 3.jpg', 'image 4.jpg',
                'image 5.jpg', 'image 6.jpg'))

#Save the location info
MODEL = 'ssd_resnet101_v1_fpn_640x640_coco17_tpu-8'
MODEL_NAME = MODEL
MODELS_DIR = os.path.join(Current_directory, 'Model')
print("The model directory is "+ MODELS_DIR)

# Save the checkpoint path
PATH_TO_CKPT = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'checkpoint\\'))
print("The check point directory is "+ PATH_TO_CKPT)

# Save the configuration path
PATH_TO_CFG = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'pipeline.config'))
print("The Configuration file is "+ PATH_TO_CFG)

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

start_time = time.time()
@tf.function
def detect_fn(image):
    """Detect objects in image."""
    
    change_image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(change_image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])
    # return detections

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# Create the category index
#PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))
PATH_TO_LABELS = 'C:/Users/anjuw/.keras/datasets/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

for i in filenames:
    #Read from the test files
    image_np = cv2.imread(Current_directory + i)
    print("Showing "+i)

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    #Detect the images 
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    #Display the image
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

    #Display final output
    cv2.imshow(i, cv2.resize(image_np_with_detections, (800, 600)))
    print("Finish Showing")
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break 

cv2.destroyAllWindows