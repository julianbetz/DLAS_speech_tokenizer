# -*- coding: utf-8 -*-

# Filename: model.py
# Author: Julian Betz and Florian Schneider
# Created: 2019-01-10
# Version: 2019-01-13
#
# Description:
#     Estimator architecture.

import tensorflow as tf


def model_fn(features, labels, mode, params):
    """
    Build the model. That is a Bi-LSTM
    :param features:
    :param labels:
    :param mode:
    :param params:
    :return:
    """

    feature_vectors, seq_lens = features['features'], features['length']

    dropout = params['dropout']
    # transpose from batch-major to time-major -> prerequisite for LSTMBlockFusedCell
    t = tf.transpose(feature_vectors, perm=[1, 0, 2])

    # print("feature_vectors.shape: %s" % feature_vectors.shape)
    # print("seq_lens.shape: %s" % seq_lens.shape)
    # print("labels.shape: %s" % labels.shape)

    # Bi-LSTM
    lstm_size = params['lstm_size']
    # LSTMBlockFusedCell is just a very efficient implementation of an LSTM Cell with correct dropout
    # i.e. an implementation of https://arxiv.org/abs/1409.2329
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(lstm_size)  # forward cell
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(lstm_size)  # backward cell
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)  # time reversed rnn since it is the backward cell
    output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=seq_lens)  # forward output
    output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=seq_lens)  # backward output
    output = tf.concat([output_fw, output_bw], axis=-1)  # concat the outputs
    output = tf.transpose(output, perm=[1, 0, 2])  # transpose back to batch-major

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    output = tf.layers.dropout(output, rate=dropout, training=is_training)  # dropout layer

    # TODO think of CRF - does it make sense for binary classification?!
    # output projection layer
    # 1 output neurons since we have binary classification. No sigmoid activation since we do this in loss function..
    logits = tf.layers.dense(output, 1, activation=None)

    # predictions are the thresholds of sigmoid of logits
    preds = tf.sigmoid(logits)
    treshed_preds = tf.round(preds)

    # print("output.shape: %s" % output.shape)
    # print("logits.shape: %s" % logits.shape)
    # print("preds.shape: %s" % preds.shape)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions
        predictions = {
            'preds': treshed_preds,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Loss function is binary (sigmoid) cross entropy
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits[:, :, 0]))

    seq_len_mask = tf.sequence_mask(seq_lens)  # to mask out the padded values
    metrics = {
        'acc': tf.metrics.accuracy(labels, treshed_preds[:, :, 0], seq_len_mask),
        'precision': tf.metrics.precision(labels, treshed_preds[:, :, 0], weights=seq_len_mask),
        'recall': tf.metrics.recall(labels, treshed_preds[:, :, 0], weights=seq_len_mask),
        'f1': tf.contrib.metrics.f1_score(labels, treshed_preds[:, :, 0], seq_len_mask)
    }
    for metric_name, op in metrics.items():
        tf.summary.scalar(metric_name, op[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    elif mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.train.AdamOptimizer().minimize(loss, global_step=tf.train.get_or_create_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

# model.py ends here
