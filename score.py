#!/usr/bin/env python3

import librosa
import librosa.display
import sys
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

def score(songname="moonlight.mp3"):
    song, sample_rate = librosa.load(songname, sr=44100)

    mel = librosa.feature.melspectrogram(song, sr=sample_rate, n_mels=128)

    log_mel = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(12,4))

    librosa.display.specshow(log_mel, sr=sample_rate, x_axis='time', y_axis='mel')

    plt.colorbar(format='%+02.0f dB')

    plt.show()

def tensorTrain(train_data, train_labels, clf):
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    
    clf.train(
        input_fn=train_input_fn,
        steps=1,
        hooks=[logging_hook])

def tensorEval(eval_data, eval_labels, clf):
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    return clf.evaluate(input_fn=eval_input_fn)


def tensorDemo():
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
   
    clf = tf.estimator.Estimator(
        model_fn=cnn, model_dir="tmp/coolModel")

    tensorTrain(train_data, train_labels, clf)

    results = tensorEval(eval_data, eval_labels, clf)

    print(results)

def cnn(features, labels, mode):
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5,5],
        padding='same',
        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5,5],
        padding='same',
        activation=tf.nn.relu
    )

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        score()
    else:
        score(sys.argv[1])
    tensorDemo()