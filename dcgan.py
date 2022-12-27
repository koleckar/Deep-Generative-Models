#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Any, Dict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--dataset", default="mnist", type=str, help="MNIST-like dataset to use.")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--train_size", default=200, type=int, help="Limit on the train set size.")
parser.add_argument("--z_dim", default=100, type=int, help="Dimension of Z.")


# If you add more arguments, ReCodEx will keep them with your default values.


# The GAN model
class GAN(tf.keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()

        self._seed = args.seed
        self._z_dim = args.z_dim
        self._z_prior = tfp.distributions.Normal(tf.zeros(args.z_dim), tf.ones(args.z_dim))

        # TODO: Define `self.generator` as a `tf.keras.Model`, which
        # - takes vectors of shape `[args.z_dim]` on input
        # - applies batch normalized dense layer with 1024 units and ReLU
        #   (do not forget about `use_bias=False` before every batch normalization)
        # - applies batch normalized dense layer with `MNIST.H // 4 * MNIST.W // 4 * 64` units and ReLU
        # - reshapes the current hidden output to `[MNIST.H // 4, MNIST.W // 4, 64]`
        # - applies batch normalized transposed convolution with 32 filters, kernel size 4,
        #   stride 2, same padding, and ReLU activation (again `use_bias=False`)
        # - applies transposed convolution with `MNIST.C` filters, kernel size 4,
        #   stride 2, same padding, and a suitable output activation
        inputs = tf.keras.layers.Input([args.z_dim])

        hidden_1 = tf.keras.layers.ReLU()(
            tf.keras.layers.BatchNormalization()(
                tf.keras.layers.Dense(units=1024, activation=None, use_bias=False)(inputs)))

        hidden_2 = tf.keras.layers.ReLU()(
            tf.keras.layers.BatchNormalization()(
                tf.keras.layers.Dense(units=MNIST.H // 4 * MNIST.W // 4 * 64, activation=None, use_bias=False)(
                    hidden_1)))

        reshaped = tf.keras.layers.Reshape(target_shape=[MNIST.H // 4, MNIST.W // 4, 64])(hidden_2)

        cnv_1 = tf.keras.layers.ReLU()(
            tf.keras.layers.BatchNormalization()(
                tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='same',
                                                activation=None, use_bias=False)(reshaped)))

        output = tf.keras.layers.Conv2DTranspose(filters=MNIST.C, kernel_size=4, strides=2,
                                                 padding='same', activation='sigmoid')(cnv_1)

        self.generator = tf.keras.Model(inputs=inputs, outputs=output, name='Generator')

        # TODO: Define `self.discriminator` as a `tf.keras.Model`, which
        # - takes input images with shape `[MNIST.H, MNIST.W, MNIST.C]`
        # - computes batch normalized convolution with 32 filters, kernel size 5,
        #   same padding, and ReLU activation (again, do not forget about
        #   `use_bias=False` before every batch normalization).
        # - max-pools with pool size 2 and stride 2
        # - computes batch normalized convolution with 64 filters, kernel size 5,
        #   same padding, and ReLU activation
        # - max-pools with pool size 2 and stride 2
        # - flattens the current representation
        # - applies batch normalized dense layer with 1024 units and ReLU activation
        # - applies output dense layer with one output and a suitable activation function
        inputs = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])

        cnv_1 = tf.keras.layers.ReLU()(
            tf.keras.layers.BatchNormalization()(
                tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation=None, use_bias=False)(
                    inputs)))
        max_pool_1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(cnv_1)

        cnv_2 = tf.keras.layers.ReLU()(
            tf.keras.layers.BatchNormalization()(
                tf.keras.layers.Conv2D(filters=64, kernel_size=5, padding='same', activation=None, use_bias=False)(
                    max_pool_1)))
        max_pool_2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(cnv_2)

        flatten = tf.keras.layers.Flatten()(max_pool_2)

        dense_1 = tf.keras.layers.ReLU()(
            tf.keras.layers.BatchNormalization()(
                tf.keras.layers.Dense(units=1024, activation=None, use_bias=False)(flatten)))

        output = tf.keras.layers.Dense(units=1, activation='sigmoid')(dense_1)

        self.discriminator = tf.keras.Model(inputs=inputs, outputs=output, name='discriminator')

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    # We override `compile`, because we want to use two optimizers.
    def compile(
            self, discriminator_optimizer: tf.optimizers.Optimizer, generator_optimizer: tf.optimizers.Optimizer
    ) -> None:
        super().compile(
            loss=tf.losses.BinaryCrossentropy(),
            metrics=tf.metrics.BinaryAccuracy("discriminator_accuracy"),
        )

        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer

    def train_step(self, images: tf.Tensor) -> Dict[str, tf.Tensor]:
        # TODO(gan): Generator training. With a Gradient tape:
        # - generate as many random latent samples as there are `images`, by a single call
        #   to `self._z_prior.sample`; also pass `seed=self._seed` for replicability;
        # - pass the samples through a generator; do not forget about `training=True`
        # - run discriminator on the generated images, also using `training=True` (even if
        #   not updating discriminator parameters, we want to perform possible BatchNorm in it)
        # - compute `generator_loss` using `self.compiled_loss`, with ones as target labels
        #   (`tf.ones_like` might come handy).
        # Then, run an optimizer step with respect to generator trainable variables.
        # Do not forget that we created generator_optimizer in the `compile` override.
        with tf.GradientTape() as tape:
            latent_samples = self._z_prior.sample(sample_shape=[tf.shape(images)[0]], seed=self._seed)
            images_fake = self.generator(latent_samples, training=True)
            discriminated_fake_images = self.discriminator(images_fake, training=True)
            generator_loss = self.compiled_loss(tf.ones_like(discriminated_fake_images), discriminated_fake_images)
        gradients = tape.gradient(generator_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        # TODO(gan): Discriminator training. Using a Gradient tape:
        # - discriminate `images` with `training=True`, storing
        #   results in `discriminated_real`
        # - discriminate images generated in generator training with `training=True`,
        #   storing results in `discriminated_fake`
        # - compute `discriminator_loss` by summing
        #   - `self.compiled_loss` on `discriminated_real` with suitable targets,
        #   - `self.compiled_loss` on `discriminated_fake` with suitable targets.
        # Then, run an optimizer step with respect to discriminator trainable variables.
        # Do not forget that we created discriminator_optimizer in the `compile` override.
        with tf.GradientTape() as tape:
            discriminated_real = self.discriminator(images, training=True)
            discriminated_fake = self.discriminator(images_fake, training=True)

            discriminator_loss_real = self.compiled_loss(tf.ones_like(discriminated_real), discriminated_real)
            discriminator_loss_fake = self.compiled_loss(tf.zeros_like(discriminated_fake), discriminated_fake)
            discriminator_loss = discriminator_loss_fake + discriminator_loss_real

        gradients = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        # TODO(gan): Update the discriminator accuracy metric -- call the
        # `self.compiled_metrics.update_state` twice, with the same arguments
        # the `self.compiled_loss` was called during discriminator loss computation.
        self.compiled_metrics.update_state(tf.ones_like(discriminated_real), discriminated_real)
        self.compiled_metrics.update_state(tf.zeros_like(discriminated_fake), discriminated_fake)

        return {
            "discriminator_loss": discriminator_loss,
            "generator_loss": generator_loss,
            **{metric.name: metric.result() for metric in self.metrics},
        }

    def generate(self, epoch: int, logs: Dict[str, tf.Tensor]) -> None:
        GRID = 20

        # Generate GRIDxGRID images
        random_images = self.generator(self._z_prior.sample(GRID * GRID, seed=self._seed), training=False)

        # Generate GRIDxGRID interpolated images
        if self._z_dim == 2:
            # Use 2D grid for sampled Z
            starts = tf.stack([-2 * tf.ones(GRID), tf.linspace(-2., 2., GRID)], -1)
            ends = tf.stack([2 * tf.ones(GRID), tf.linspace(-2., 2., GRID)], -1)
        else:
            # Generate random Z
            starts = self._z_prior.sample(GRID, seed=self._seed)
            ends = self._z_prior.sample(GRID, seed=self._seed)
        interpolated_z = tf.concat(
            [starts[i] + (ends[i] - starts[i]) * tf.linspace(0., 1., GRID)[:, tf.newaxis] for i in range(GRID)],
            axis=0)
        interpolated_images = self.generator(interpolated_z, training=False)

        # Stack the random images, then an empty row, and finally interpolated images
        image = tf.concat(
            [tf.concat(list(images), axis=1) for images in tf.split(random_images, GRID)] +
            [tf.zeros([MNIST.H, MNIST.W * GRID, MNIST.C])] +
            [tf.concat(list(images), axis=1) for images in tf.split(interpolated_images, GRID)], axis=0)
        with self.tb_callback._train_writer.as_default(step=epoch):
            tf.summary.image("images", image[tf.newaxis])


def main(args: argparse.Namespace) -> Dict[str, float]:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load data
    mnist = MNIST(args.dataset, size={"train": args.train_size})
    train = mnist.train.dataset.map(lambda example: example["images"])
    train = train.shuffle(mnist.train.size, args.seed)
    train = train.batch(args.batch_size)

    # Create the network and train
    network = GAN(args)
    network.compile(
        discriminator_optimizer=tf.optimizers.Adam(),
        generator_optimizer=tf.optimizers.Adam(),
    )
    logs = network.fit(train, epochs=args.epochs, callbacks=[
        tf.keras.callbacks.LambdaCallback(on_epoch_end=network.generate), network.tb_callback])

    # Return loss and discriminator accuracy for ReCodEx to validate
    return {metric: logs.history[metric][-1] for metric in ["loss", "discriminator_accuracy"]}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
