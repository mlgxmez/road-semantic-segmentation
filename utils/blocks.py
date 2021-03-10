import tensorflow as tf
from tensorflow.keras import layers, regularizers

NUM_CLASSES = 2


def conv1x1(inputs):
    return (layers.Conv2D(
        NUM_CLASSES,
        1,
        padding="same",
        kernel_regularizer=regularizers.l2(1e-3))(inputs)
    )


def build_decoder(pool3, pool4, pool5):

    conv1x1_out3 = conv1x1(pool3.output)
    conv1x1_out4 = conv1x1(pool4.output)
    conv1x1_out5 = conv1x1(pool5.output)

    output = (layers
              .Conv2DTranspose(
                  NUM_CLASSES,
                  4,
                  2,
                  padding="same",
                  kernel_regularizer=regularizers.l2(1e-3))(conv1x1_out5)
              )

    output = tf.add(output, conv1x1_out4)

    output = (layers
              .Conv2DTranspose(
                  NUM_CLASSES,
                  4,
                  2,
                  padding="same",
                  kernel_regularizer=regularizers.l2(1e-3))(output)
              )

    output = tf.add(output, conv1x1_out3)

    output = (layers
              .Conv2DTranspose(
                  NUM_CLASSES,
                  16,
                  8,
                  padding="same",
                  kernel_regularizer=regularizers.l2(1e-3))(output)
              )

    return output
