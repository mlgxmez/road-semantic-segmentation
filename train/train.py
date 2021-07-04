import argparse

import tensorflow as tf
from tensorflow import keras
from azureml.core import Model, Run

from augment import Augmentation

parser = argparse.ArgumentParser()

parser.add_argument('--nontrained_model')
parser.add_argument('--trained_model')
parser.add_argument('--input_train_step')
parser.add_argument('--output_train_step')
parser.add_argument('--augmentation_file')

args = parser.parse_args()

PATH_IMAGES = args.input_train_step
PATH_MASKS = args.output_train_step
AUGMENTATION_FILE = args.augmentation_file
SEED = 1234

# For debugging purposes only
print("--trained_model:", args.trained_model)
print("--input_train_step:", args.input_train_step)
print("--output_train_step:", args.output_train_step)


@tf.function
def binarize_mask(mask_img):
    # All pixels not labeled as road will be 1, else 0
    mask = mask_img[:, :, 0]
    mask = tf.where(mask == 0, 1, 0)
    return mask[..., tf.newaxis]


aug = Augmentation(AUGMENTATION_FILE)
generator_config_images = dict(
    **aug.config,
    rescale=1./255)
generator_config_masks = dict(
    **aug.config,
    preprocessing_function=binarize_mask)

image_datagen = (keras
                 .preprocessing
                 .image
                 .ImageDataGenerator(**generator_config_images))

mask_datagen = (keras
                .preprocessing
                .image
                .ImageDataGenerator(**generator_config_masks))

image_generator = (image_datagen
                   .flow_from_directory(PATH_IMAGES,
                                        target_size=(224, 224),
                                        class_mode=None,
                                        seed=SEED))

mask_generator = (mask_datagen
                  .flow_from_directory(PATH_MASKS,
                                       target_size=(224, 224),
                                       class_mode=None,
                                       seed=SEED))

# Download the model architecture from AzureML
run = Run.get_context()
ws = run.experiment.workspace
path_model = Model.get_model_path(
    "segmentation_new",
    version=2,
    _workspace=ws)

# Load and train the model
loaded_model = keras.models.load_model(path_model)
train_generator = zip(image_generator, mask_generator)
loaded_model.fit(train_generator,
                 epochs=3,
                 steps_per_epoch=100)

# Save the model in the path specified of the compute target
loaded_model.save(args.trained_model)
