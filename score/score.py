import argparse
import os
import glob
import time

import tensorflow as tf
from tensorflow import keras
from azureml.core import Model, Run

# TODO: Parse arguments coming from the Pipeline step
# TODO: Scoring images and labels plus path to the model in datastore

parser = argparse.ArgumentParser()
parser.add_argument('--trained_model')
parser.add_argument('--input_score_step')
parser.add_argument('--output_score_step')

args = parser.parse_args()

MODEL_NAME = 'road-segmentation'
PATH_MODEL = args.trained_model
MOUNT_POINT_IMAGES = args.input_score_step
MOUNT_POINT_LABELS = args.output_score_step

loaded_model = keras.models.load_model(args.trained_model)

eval_input_files = glob.glob(os.path.join(MOUNT_POINT_IMAGES, '**/*.png'))
eval_output_files = glob.glob(os.path.join(MOUNT_POINT_LABELS, '**/*.png'))


@tf.function
def preprocessEvalImage(image_file):
    image_bytes = tf.io.read_file(image_file)
    image_tensor = tf.io.decode_png(image_bytes)
    image_resized = tf.image.resize(
        image_tensor,
        size=[224, 224],
        method='nearest')
    return image_resized


@tf.function
def binarizeMask(mask_img):
    # All pixels not labeled as road will be 1, else 0
    mask = mask_img[:, :, 0]
    mask = tf.where(mask == 0, 1, 0)
    return mask[..., tf.newaxis]


score = []
total_time = 0.0

for img_file, mask_file in zip(eval_input_files, eval_output_files):
    input_tensor = preprocessEvalImage(img_file)
    output_tensor = binarizeMask(preprocessEvalImage(mask_file))
    start_time = time.time()
    loss, acc = loaded_model.evaluate(
        input_tensor[tf.newaxis, ...],
        output_tensor[tf.newaxis, ...])
    eval_time = time.time() - start_time
    total_time += eval_time
    score.append(acc)

avg_score = sum(score)/len(score)
avg_time = total_time/len(score)

tags = {
    'accuracy': avg_score,
    'latency': avg_time,
    'num_images': len(score),
    'name': 'eval'
    }

# Register the model only if it has highest accuracy among the trained models
run = Run.get_context()
ws = run.experiment.workspace

registered_models = Model.list(ws, name=MODEL_NAME)
max_accuracy = 0
for model in registered_models:
    registered_accuracy = model.tags['accuracy']
    if max_accuracy < registered_accuracy:
        max_accuracy = registered_accuracy

if max_accuracy < avg_score:
    # TODO: Verify that images and masks are splitted in the same way
    Model.register(
        workspace=ws,
        model_path=PATH_MODEL,  # model_path contains architecture and weights
        model_name=MODEL_NAME,
        tags=tags,
        model_framework=Model.Framework.TENSORFLOW,
        model_framework_version='2.3')  # Second element is type Dataset
