import tensorflow as tf
from .blocks import ResidualBlock, VectorQuantizationLayer
from tensorflow.keras import layers

def get_encoder(input_shape, initial_dim=64, embed_dim=16, depths=[2, 2, 2, 3]):

    input = layers.Input(shape=input_shape, name="Input")
    x = layers.Conv2D(
        initial_dim, kernel_size=3, strides=1, padding="same", name="InitialConv"
    )(input)

    for i in range(len(depths)):

        factor = min((2**i), 8)

        for j in range(depths[i]):
            x = ResidualBlock(
                initial_dim * factor, name=f"layer_{i}_ResidualBlock_{j}"
            )(x)

        if i < len(depths) - 1:
            x = layers.Conv2D(
                initial_dim * factor,
                kernel_size=3,
                strides=2,
                padding="same",
                name=f"layer_{i}_DownSampling",
            )(x)

    x = layers.LayerNormalization(epsilon=1e-6, name="FinalLN")(x)
    x = layers.Activation(activation="linear", name="FinalAct")(x)

    x = layers.Conv2D(embed_dim, kernel_size=3, padding="same", name="FinalConv")(x)
    out = layers.Conv2D(embed_dim, kernel_size=1, name="Embedding")(x)

    encoder = tf.keras.models.Model(inputs=input, outputs=out)

    return encoder


def get_decoder(input_shape, initial_dim=512, depths=[3, 2, 2, 2]):

    input = layers.Input(shape=input_shape, name="Input")
    x = layers.Conv2D(
        input.shape[-1], kernel_size=1, strides=1, padding="same", name="InitialConv"
    )(input)

    for i in range(len(depths)):

        factor = min((2**i), 8)

        for j in range(depths[i]):
            x = ResidualBlock(
                initial_dim // factor, name=f"layer_{i}_ResidualBlock_{j}"
            )(x)

        if i < len(depths) - 1:
            x = layers.UpSampling2D(interpolation="bilinear")(x)

    x = layers.LayerNormalization(epsilon=1e-6, name="FinalLN")(x)
    x = layers.Activation(activation="linear", name="FinalAct")(x)

    # x = layers.Conv2D(16, kernel_size=3, padding='same', name='FinalConv')(x)
    out = layers.Conv2D(3, kernel_size=1, name="Embedding", dtype=tf.float32)(x)

    decoder = tf.keras.models.Model(inputs=input, outputs=out)

    return decoder


def VQVAE(
    input_shape, embed_dim=8, num_vectors=256, initial_dim=64, depths=[2, 2, 2, 3]
):
    """Instantiates a VQVAE model. https://arxiv.org/abs/1711.00937

    Args:
        input_shape (tuple): The shape of the input tensor.
        embed_dim (int, optional): The number of channels of the encoder embedding.
            This will be the dimension of the encoding vectors of the codebook. Defaults to 8.
        num_vectors (int, optional): The number of encoding vectors for the codebook.
            Defaults to 256.
        initial_dim (int, optional): The initial embedding dim.
            The dimension of successive layers will be initial_dim * 2**i,
            where i is the index of the layer. Saturates to initial_dim * 2**(n-2),
            where n is the number of layers. Defaults to 64.
        depths (list, optional): The depth of each layer. Each layer i will have
            depths[i] blocks. Defaults to [2, 2, 2, 3].

    Returns:
        _type_: _description_
    """

    vqlayer = VectorQuantizationLayer(
        num_vectors=num_vectors, vector_dimension=embed_dim, name="vector_quantizer"
    )
    H, W, C = input_shape

    dw_factor = 2 ** (len(depths) - 1)
    encoder = get_encoder(
        input_shape=input_shape,
        initial_dim=initial_dim,
        embed_dim=embed_dim,
        depths=depths,
    )
    depths.reverse()
    decoder = get_decoder(
        input_shape=(H // dw_factor, W // dw_factor, embed_dim),
        initial_dim=initial_dim * (dw_factor / 2),
        depths=depths,
    )

    inputs = layers.Input(shape=input_shape)
    embedding = encoder(inputs)
    quantized_embedding = vqlayer(embedding)
    reconstruction = decoder(quantized_embedding)

    vqvae = tf.keras.models.Model(inputs=inputs, outputs=reconstruction, name="VQ-VAE")

    return vqvae
