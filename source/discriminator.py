import tensorflow as tf
from tensorflow.keras import layers


def get_discriminator(
    input_shape=(256, 256, 3), num_layers=3, kernel_size=4, initial_filters=64
):

    inputs = layers.Input(shape=input_shape, name="Disc_Input")

    x = layers.Conv2D(
        filters=initial_filters,
        kernel_size=kernel_size,
        strides=2,
        padding="same",
        name="Disc_conv",
    )(inputs)
    x = layers.BatchNormalization(name="Disc_bn")(x)
    x = layers.Activation("gelu", name="Disc_act")(x)

    for i in range(1, num_layers, 1):

        factor = 2**i

        x = layers.Conv2D(
            filters=int(initial_filters * factor),
            kernel_size=kernel_size,
            strides=2,
            padding="same",
            name=f"Disc_conv_{i}",
        )(x)
        x = layers.BatchNormalization(name=f"Disc_bn_{i}")(x)
        x = layers.Activation("gelu", name=f"Disc_act_{i}")(x)

    final_conv = layers.Conv2D(
        filters=1, kernel_size=1, strides=1, name="Disc_final_conv"
    )(x)

    discriminator = tf.keras.models.Model(
        inputs=inputs, outputs=final_conv, name="discriminator"
    )

    return discriminator
