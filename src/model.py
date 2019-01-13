# -*- coding: utf-8 -*-

# Filename: model.py
# Author: Julian Betz
# Created: 2019-01-10
# Version: 2019-01-13
#
# Description:
#     Estimator architecture.

import numpy as np
import tensorflow as tf
import pandas as pd

def model_fn(features, labels, mode, params): # TODO
    # print('<Called model function>')

    # Directly access features/labels since feature column implementations for sequences are still experimental
    # TODO@Flo: The following (commented) lines are the proper way to access the features for the dynamic RNN
    # net = features['features']          # Input layer in batches
    # length = features['length']         # Sequence lengths

    # TODO@Flo: The following lines are only temporary (the below network is currently not adapted to sequences)
    net = features['features'][:, 0]
    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = tf.cast(labels[:, 0], tf.int64)

    # net = tf.constant(0.5, shape=(BATCH_SIZE, N_FEATURES), dtype=tf.float32)
    # labels = tf.constant(0, shape=(BATCH_SIZE,), dtype=tf.int64)

    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    logits = tf.layers.dense(net, units=params['n_classes'], activation=None)
    predicted_classes = tf.argmax(logits, axis=1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids' : predicted_classes[:, tf.newaxis],
            'probabilities' : tf.nn.softmax(logits),
            'logits' : logits}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')
    metrics = {'accuracy' : accuracy}
    tf.summary.scalar('accuracy', accuracy[1])
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

# model.py ends here
