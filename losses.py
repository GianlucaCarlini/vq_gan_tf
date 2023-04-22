import tensorflow as tf


def discriminator_loss(real_output, generated_output):

    ones = tf.ones_like(real_output)
    zeros = tf.zeros_like(generated_output)

    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(ones, real_output)
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        zeros, generated_output
    )

    return real_loss + generated_loss

def vq_vae_loss(y_true, y_pred):

    loss = tf.reduce_mean(tf.abs(y_true - y_pred))

    return loss