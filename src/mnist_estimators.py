""" CNN - MNIST example

Example of use of Tensorflow estimators (high level interface) with the MNIST dataset

Mostly used for experimenting with the different kinds of layers.

It was also useful for testing the running time improvement when using a native build.

For future reference:

Tensorflow 1.5 (Feb'18) (a.k.a pip install tensorflow): 89 minutes - 225 steps/m
Tensorflow 1.6 (-march=native):                         55 minutes - 365 steps/m

On a MacBook Pro (Early 2015) I7 3.1 Ghz, 16 GB Ram, CPU mode.

Usage:  python mnist_estimators.py --model_path <path_to_store_model>
"""

from argparse import ArgumentParser

import numpy as np
import tensorflow as tf


def cnn_model(features, labels, mode):
    """

    Args:
        features (Tensor): Input features from data - likely from a single batch
        labels (Tensor): Output feature (class), in one-hot format
        mode (tf.estimator.ModeKeys): Current mode of the estimator

    Returns:
        tf.estimator.EstimatorSpec, representing the current state of the network
    """

    # note the current stage of the model
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    is_prediction = mode == tf.estimator.ModeKeys.PREDICT
    is_evaluation = mode == tf.estimator.ModeKeys.EVAL

    # input layer   (output is #n_images, height=28, width=28, channels=1)
    input_data = tf.reshape(features["x"], [-1, 28, 28, 1])

    # first convolutional layer (output is #n_images, height=28, width=28, filters=32)
    conv1 = tf.layers.conv2d(
        inputs=input_data,
        filters=32,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu)

    # first pooling (output is #n_images, height=14, width=14, filters=32)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)

    # second convolutional layer (output is #n_images, height=14, width=14, filters=64)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu)

    # second pooling (output is #n_images, height=7, width=7, filters=64)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)

    # dense layer + dropout (output is #n_images, 1024)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=is_training)

    # logits layer (output is #n_images, classes=10)
    logits = tf.layers.dense(inputs=dropout, units=10)

    # we are interested in both the classes and the probabilities
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax")
    }

    if mode == is_prediction:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # loss function
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if is_training:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    if is_evaluation:
        eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels,
                                                           predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    return None


def main(parameters):

    # load the MNIST dataset - this will raise a bunch of DeprecationWarnings
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # define the network as an estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model,
                                              model_dir=parameters['model_path'])

    # define how the training data will be presented
    train_function = tf.estimator.inputs.numpy_input_fn(x={"x": train_data}, y=train_labels,
                                                        batch_size=100, num_epochs=None,
                                                        shuffle=True)
    # train the model
    mnist_classifier.train(input_fn=train_function, steps=20000)

    # define how the validation data will be presented
    eval_function = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, y=eval_labels,
                                                       num_epochs=1, shuffle=False)
    # evaluate the model
    eval_results = mnist_classifier.evaluate(input_fn=eval_function)

    print(eval_results)

if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('--model_path', required=True)

    settings = vars(parser.parse_args())

    main(settings)
