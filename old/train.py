from getFeatures import getFeatures
import json
import numpy as np
import tensorflow as tf

# Trains Tensorflow using wav files and json labels
def train(features):

    train_data = None
    for i in range(10):
        feature, tempo = getFeatures("alda-ml/samples/" + str(i+1) + "/out.wav")
        feature = np.reshape(feature[-1], -1).astype('float32')
        if train_data is not None:
            train_data = np.vstack([train_data, feature])
        else:
            train_data = np.array(feature)
    # print(train_data.shape)

    train_labels = None
    for i in range(10):
        with open("alda-ml/samples/" + str(i+1) + "/score.json") as f:
            score = json.loads(f.read())
            note = score['events'][0]['midi-note']
            label = [0 for x in range(88)]
            label[note] = 1
        if train_labels is not None:
            train_labels = np.vstack([train_labels, label])
        else:
            train_labels = np.array(label)
    # print(train_labels.shape)

    clf = tf.estimator.Estimator(
        model_fn=cnn, model_dir="tmp/coolModel")

    tensorTrain(train_data, train_labels, clf)

    return clf

def tensorTrain(train_data, train_labels, clf):
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=10,
        num_epochs=None,
        shuffle=True)
    
    clf.train(
        input_fn=train_input_fn,
        steps=1)

def tensorEval(eval_data, eval_labels, clf):
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    return clf.evaluate(input_fn=eval_input_fn)

def cnn(features, labels, mode):
    input_layer = tf.reshape(features['x'], [-1, 48, 336, 1])

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

    pool2_flat = tf.reshape(pool2, [-1, 12 * 84 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=88)

    predictions = {
        "classes": logits,
        "probabilities": logits
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.mean_squared_error(
        labels=labels, predictions=logits)

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
