import tensorflow as tf
from .blocks import ResidualBlock, VectorQuantizationLayer
from tensorflow.keras import layers


def get_encoder(input_shape, initial_dim=64, embed_dim=16, depths=[2, 2, 2, 3]):

    input = layers.Input(shape=input_shape, name="Encoder_Input")
    x = layers.Conv2D(
        initial_dim,
        kernel_size=3,
        strides=1,
        padding="same",
        name="Encoder_InitialConv",
    )(input)

    for i in range(len(depths)):

        factor = min((2**i), 8)

        for j in range(depths[i]):
            x = ResidualBlock(
                initial_dim * factor, name=f"Encoder_layer_{i}_ResidualBlock_{j}"
            )(x)

        if i < len(depths) - 1:
            x = layers.Conv2D(
                initial_dim * factor,
                kernel_size=3,
                strides=2,
                padding="same",
                name=f"Encoder_layer_{i}_DownSampling",
            )(x)

    x = layers.LayerNormalization(epsilon=1e-6, name="Encoder_FinalLN")(x)
    x = layers.Activation(activation="linear", name="Encoder_FinalAct")(x)

    x = layers.Conv2D(
        embed_dim, kernel_size=3, padding="same", name="Encoder_FinalConv"
    )(x)
    out = layers.Conv2D(embed_dim, kernel_size=1, name="Encoder_Embedding")(x)

    encoder = tf.keras.models.Model(inputs=input, outputs=out, name="Encoder")

    return encoder


def get_decoder(input_shape, initial_dim=512, depths=[3, 2, 2, 2]):

    input = layers.Input(shape=input_shape, name="Decoder_Input")
    x = layers.Conv2D(
        input.shape[-1],
        kernel_size=1,
        strides=1,
        padding="same",
        name="Decoder_InitialConv",
    )(input)

    for i in range(len(depths)):

        factor = min((2**i), 8)

        for j in range(depths[i]):
            x = ResidualBlock(
                initial_dim // factor, name=f"Decoder_layer_{i}_ResidualBlock_{j}"
            )(x)

        if i < len(depths) - 1:
            x = layers.UpSampling2D(
                interpolation="bilinear", name=f"Decoder_layer{i}_Upsampling{j}"
            )(x)

    x = layers.LayerNormalization(epsilon=1e-6, name="Decoder_FinalLN")(x)
    x = layers.Activation(activation="linear", name="Decoder_FinalAct")(x)

    # x = layers.Conv2D(16, kernel_size=3, padding='same', name='FinalConv')(x)
    out = layers.Conv2D(3, kernel_size=1, name="Decoder_Embedding", dtype=tf.float32)(x)

    decoder = tf.keras.models.Model(inputs=input, outputs=out, name="Decoder")

    return decoder


class VQ_VAE(tf.keras.models.Model):
    def __init__(
        self,
        input_shape,
        embed_dim=8,
        num_vectors=256,
        initial_dim=64,
        depths=[2, 2, 2, 3],
        *args,
        **kwargs,
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

        super().__init__(*args, **kwargs)

        self._input_shape = input_shape
        self.embed_dim = embed_dim
        self.num_vectors = num_vectors
        self.initial_dim = initial_dim
        self.depths = depths
        self.ds_factor = 2 ** (len(depths) - 1)

        self.encoder = get_encoder(
            input_shape=self._input_shape,
            initial_dim=self.initial_dim,
            embed_dim=self.embed_dim,
            depths=self.depths,
        )

        self.vq_layer = VectorQuantizationLayer(
            num_vectors=self.num_vectors,
            vector_dimension=self.embed_dim,
            name="vector_quantizer",
        )

        depths.reverse()
        self.reversed = depths

        H, W, _ = self._input_shape
        self.decoder = get_decoder(
            input_shape=(H // self.ds_factor, W // self.ds_factor, self.embed_dim),
            initial_dim=self.initial_dim * self.ds_factor,
            depths=self.reversed,
        )

    def call(self, inputs):

        x = self.encoder(inputs)
        x = self.vq_layer(x)
        x = self.decoder(x)

        return x
