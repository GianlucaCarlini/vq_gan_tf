import tensorflow as tf
from .vq_vae import VQ_VAE
from .losses import discriminator_loss, generator_loss, l1_loss
from .discriminator import get_discriminator
from .utils import WarmUp

__all__ = ["VQ_Gan"]


class VQ_Gan(tf.keras.models.Model):
    def __init__(
        self, input_shape, gen_optimizer=None, disc_optimizer=None, *args, **kwargs
    ):
        super().__init__()

        self._input_shape = input_shape
        self.embed_dim = kwargs.get("embed_dim", 8)
        self.num_vectors = kwargs.get("num_vectors", 128)
        self.initial_dim = kwargs.get("initial_dim", 64)
        self.depths = kwargs.get("depths", [2, 2, 2, 3])

        self.disc_num_layers = kwargs.get("num_layers", 3)
        self.disc_kernel_size = kwargs.get("kernel_size", 4)
        self.disc_initial_filters = kwargs.get("initial_filters", 64)

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.vq_vae_loss_tracker = tf.keras.metrics.Mean(name="vq_vae_loss")
        self.generator_loss_tracker = tf.keras.metrics.Mean(name="generator_loss")
        self.discriminator_loss_tracker = tf.keras.metrics.Mean(
            name="discriminator_loss"
        )
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )

        if gen_optimizer is not None:
            self.gen_optimizer = gen_optimizer
        else:
            self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        if disc_optimizer is not None:
            self.disc_optimizer = disc_optimizer
        else:
            self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        self.vq_vae = VQ_VAE(
            input_shape=self._input_shape,
            embed_dim=self.embed_dim,
            num_vectors=self.num_vectors,
            initial_dim=self.initial_dim,
            depths=self.depths,
        )

        self.discriminator = get_discriminator(
            input_shape=self._input_shape,
            num_layers=self.disc_num_layers,
            kernel_size=self.disc_kernel_size,
            initial_filters=self.disc_initial_filters,
        )

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.vq_vae_loss_tracker,
            self.generator_loss_tracker,
            self.discriminator_loss_tracker,
            self.reconstruction_loss_tracker,
        ]

    def train_step(self, inputs):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            reconstructions = self.vq_vae(inputs, training=True)
            disc_real_output = self.discriminator(inputs)
            disc_gen_outputs = self.discriminator(reconstructions)

            gen_loss = generator_loss(
                y_true=inputs, y_pred=reconstructions, disc_output=disc_gen_outputs
            )
            vq_vae_loss = gen_loss + sum(self.vq_vae.losses)

            disc_loss = 0.05 * discriminator_loss(disc_real_output, disc_gen_outputs)

            # only used as metric
            reconstruction_loss = l1_loss(y_true=inputs, y_pred=reconstructions)

        generator_gradients = gen_tape.gradient(
            vq_vae_loss, self.vq_vae.trainable_variables
        )
        discriminator_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_weights
        )

        self.gen_optimizer.apply_gradients(
            zip(generator_gradients, self.vq_vae.trainable_variables)
        )
        self.disc_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables)
        )

        total_loss = vq_vae_loss + disc_loss

        self.total_loss_tracker.update_state(total_loss)
        self.vq_vae_loss_tracker.update_state(sum(self.vq_vae.losses))
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.generator_loss_tracker.update_state(gen_loss)
        self.discriminator_loss_tracker.update_state(disc_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):

        reconstructions = self.vq_vae(inputs, training=True)
        disc_real_output = self.discriminator(inputs)
        disc_gen_outputs = self.discriminator(reconstructions)

        gen_loss = generator_loss(
            y_true=inputs, y_pred=reconstructions, disc_output=disc_gen_outputs
        )
        vq_vae_loss = gen_loss + sum(self.vq_vae.losses)

        disc_loss = 0.05 * discriminator_loss(disc_real_output, disc_gen_outputs)

        # only used as metric
        reconstruction_loss = l1_loss(y_true=inputs, y_pred=reconstructions)

        total_loss = vq_vae_loss + disc_loss

        self.total_loss_tracker.update_state(total_loss)
        self.vq_vae_loss_tracker.update_state(sum(self.vq_vae.losses))
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.generator_loss_tracker.update_state(gen_loss)
        self.discriminator_loss_tracker.update_state(disc_loss)

        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=False):

        x = self.vq_vae(inputs, training=training)

        return x
