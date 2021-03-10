# This script builds the model or models
from tensorflow import keras
import argparse

from utils.blocks import build_decoder

parser = argparse.ArgumentParser()
parser.add_argument('--path_model')
parser.add_argument('--plot_model', action='store_true')

args = parser.parse_args()

# Replicate the same structure as in old repo
base_model = keras.applications.VGG16(input_shape=(224, 224, 3),
                                      include_top=False)

base_model.trainable = False

# Layers from which to apply deconvolution
# Reference: "Fully Convolutional Networks for Semantic Segmentation"
layer_names = ["block3_pool", "block4_pool", "block5_pool"]

layers = [base_model.get_layer(layer) for layer in layer_names]

decoder_output = build_decoder(*layers)
model = keras.Model(base_model.inputs, decoder_output)

model.compile(optimizer="adam",
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=["accuracy"])

if args.plot_model:
    keras.utils.plot_model(
        model,
        to_file="vgg16_segmentation.png",
        show_shapes=True,
        show_layer_names=True)

model.save(args.path_model)
