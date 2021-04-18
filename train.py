import argparse

import tensorflow as tf
from tensorflow import keras

from utils.augment import Augmentation

parser = argparse.ArgumentParser()

parser.add_argument('--nontrained_model')
parser.add_argument('--trained_model')
parser.add_argument('--input_train_step')
parser.add_argument('--output_train_step')
parser.add_argument('--augmentation_file')
parser.add_argument('--local', action="store_true")

args = parser.parse_args()

if args.local and args.nontrained_model is None:
    raise ValueError("To train a model locally, "
                     "you must specify the --nontrained_model argument.")

PATH_IMAGES = args.input_train_step
PATH_MASKS = args.output_train_step
AUGMENTATION_FILE = args.augmentation_file

LOCAL = True
SEED = 1234

# For debugging purposes only
print("--trained_model:", args.trained_model)
print("--input_train_step:", args.input_train_step)
print("--output_train_step:", args.output_train_step)
print(AUGMENTATION_FILE)
print(PATH_IMAGES)
print(PATH_MASKS)


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

print("Example image dimensions: {}".format(tf.shape(image_generator.next())))
print("Example mask dimensions: {}".format(tf.shape(mask_generator.next())))

if LOCAL:
    loaded_model = keras.models.load_model(args.nontrained_model)

else:
    from azureml.core import Model, Workspace

    ws = Workspace.from_config()
    path_model = Model.get_model_path(
        "segmentation_new",
        version=1,
        _workspace=ws)

    loaded_model = keras.models.load_model(path_model)

train_generator = zip(image_generator, mask_generator)
loaded_model.fit(train_generator,
                 steps_per_epoch=100,
                 epochs=10)

loaded_model.save(args.trained_model)
