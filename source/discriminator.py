import tensorflow as tf
from tensorflow.keras import layers


def get_discriminator(
    input_shape=(256, 256, 3), num_layers=3, kernel_size=4, initial_filters=64
):

    inputs = layers.Input(shape=input_shape, name="Input")

    x = layers.Conv2D(
        filters=initial_filters, kernel_size=kernel_size, strides=2, padding="same"
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)

    for i in range(1, num_layers, 1):

        factor = 2**i

        x = layers.Conv2D(
            filters=int(initial_filters * factor),
            kernel_size=kernel_size,
            strides=2,
            padding="same",
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("gelu")(x)

    final_conv = layers.Conv2D(filters=1, kernel_size=1, strides=1)(x)

    discriminator = tf.keras.models.Model(
        inputs=inputs, outputs=final_conv, name="discriminator"
    )

    return discriminator
