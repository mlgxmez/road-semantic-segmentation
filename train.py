import os
import argparse

import tensorflow as tf
from tensorflow import keras

from utils.augment import Augmentation

parser = argparse.ArgumentParser()
parser.add_argument('--path_data')
parser.add_argument('--dir_images', default='images')
parser.add_argument('--dir_masks', default='labels')
parser.add_argument('--augmentation_file')
parser.add_argument('--path_model')

args = parser.parse_args()

PATH_IMAGES = os.path.join(args.path_data, args.dir_images)
PATH_MASKS = os.path.join(args.path_data, args.dir_masks)
AUGMENTATION_FILE = args.augmentation_file

PADDING_VALUE = 'same'
NUM_CLASSES = 2
SEED = 1234


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
                                       color_mode="grayscale",
                                       class_mode=None,
                                       seed=SEED))

# TODO: These lines should be added as tests, add tests for the save model
print("Example image dimensions: {}".format(tf.shape(image_generator.next())))
print("Example mask dimensions: {}".format(tf.shape(mask_generator.next())))

loaded_model = keras.models.load_model(args.path_model)

train_generator = zip(image_generator, mask_generator)
loaded_model.fit(train_generator,
                 steps_per_epoch=100,
                 epochs=10)

loaded_model.save_weights(os.path.join(os.getcwd(), args.path_model, "ckpt"))
