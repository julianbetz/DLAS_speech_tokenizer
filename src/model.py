# -*- coding: utf-8 -*-

# Filename: model.py
# Author: Julian Betz and Florian Schneider
# Created: 2019-01-10
# Version: 2019-01-24
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
    with tf.variable_scope('my_model', reuse=tf.AUTO_REUSE):
        # General parameters
        dropout = params['dropout']
        is_training = mode == tf.estimator.ModeKeys.TRAIN

        # Input features
        feature_vectors, seq_lens = features['features'], features['length']

        # Bi-LSTM
        lstm_size = params['lstm_size']
        # transpose from batch-major to time-major -> prerequisite for LSTMBlockFusedCell
        input_tensor = tf.transpose(feature_vectors, perm=[1, 0, 2])

        lstm_cell_fw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(lstm_size)  # forward cell
        lstm_cell_bw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(lstm_size)  # backward cell

        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, input_tensor, sequence_length=seq_lens, dtype=tf.float32, time_major=True)

        layer = tf.concat([outputs[0], outputs[1]], axis=-1)  # concat the outputs
        layer = tf.transpose(layer, perm=[1, 0, 2])  # transpose back to batch-major
        layer = tf.layers.dropout(layer, rate=dropout, training=is_training)  # dropout layer

        # DNN on top of the Bi-LSTM
        for n_units in params['dense_sizes']:
            layer = tf.layers.dense(layer, n_units, activation=tf.nn.relu) # Dense layer
            layer = tf.layers.dropout(layer, rate=dropout, training=is_training) # Dropout layer

        # TODO Possibly add a CRF?!

        # Output layer
        # 1 output neuron since we perform binary classification. No sigmoid activation since we apply it implicitly in the loss function.
        logits = tf.layers.dense(layer, 1, activation=None)
        # The predicted classes are the thresholded sigmoids of logits
        preds = tf.sigmoid(logits)
        treshed_preds = tf.round(preds)

        # print("feature_vectors.shape: %s" % feature_vectors.shape)
        # print("seq_lens.shape: %s" % seq_lens.shape)
        # print("labels.shape: %s" % labels.shape)
        # print("output.shape: %s" % output.shape)
        # print("logits.shape: %s" % logits.shape)
        # print("preds.shape: %s" % preds.shape)

        # Prediction
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'predictions': treshed_preds, # Class IDs
                'probabilities': preds # Interpret tendency towards class one as its probability
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # Loss
        seq_len_mask = tf.sequence_mask(seq_lens)  # to mask out the padded values
        weighted_cross_entropy = tf.nn.weighted_cross_entropy_with_logits(
            targets=labels,
            logits=logits[:, :, 0],
            pos_weight=tf.constant(27.244325949851728 / 2, tf.float32))
        loss = tf.reduce_sum(weighted_cross_entropy * tf.cast(seq_len_mask, tf.float32))
        loss = loss / tf.reduce_sum(tf.cast(seq_lens, tf.float32))

        # Metrics
        metrics = {
            'acc': tf.metrics.accuracy(labels, treshed_preds[:, :, 0], seq_len_mask),
            'precision': tf.metrics.precision(labels, treshed_preds[:, :, 0], weights=seq_len_mask),
            'recall': tf.metrics.recall(labels, treshed_preds[:, :, 0], weights=seq_len_mask),
            'f1': tf.contrib.metrics.f1_score(labels, treshed_preds[:, :, 0], seq_len_mask)
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        # Evaluation
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        # Training
        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer().minimize(loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

# model.py ends here
