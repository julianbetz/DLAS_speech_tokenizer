#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Filename: main.py
# Author: Julian Betz and Florian Schneider
# Created: 2018-12-20
# Version: 2019-01-24
#
# Description:
#     Entry point to the training/testing procedures.

import datetime
import json
import logging
import os
import re
import sys
from itertools import count
from time import time

import click
import numpy as np
import tensorflow as tf
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll import scope
from numpy.random import RandomState
from matplotlib import pyplot as plt
from pathlib2 import Path

from data_handler import DataLoader, input_fn, TRAIN_MODE, EVAL_MODE, PRED_MODE, align_seqs_to_breaking_labels
from kaldi_io import readArk
from model import model_fn
from util import progress

SEED = 3735758343
BATCH_SEED = 2068916927
# Sample input data to analyze its structure, making sure that the DateLoader is only instantiated once
N_FEATURES = [loader.load(loader.ids[0])[0].shape[1] for loader in [
    DataLoader(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../dat/fast_load'), tst_size=0,
               seed=SEED)]][0]
FEATURE_COLS = [tf.feature_column.numeric_column(key='features', shape=N_FEATURES)]

@click.command()
@click.option('--alignments/--no-alignments', '-a/', default=False, help='Whether to convert alignment data',
              show_default=True)
@click.option('--spectrograms/--no-spectrograms', '-s/', default=False, help='Whether to convert the spectrograms',
              show_default=True)
@click.option('--operation', '-o', type=click.Choice(['hyperoptimize', 'predict']),
              help='The operation to perform on the estimator')
@click.option('--model_dir', '-d', default=None, type=str, help='Where to store model data.')
@click.option('--tst_size', type=str,
              help='The number/proportion of samples from the complete dataset to reserve for testing.  [default: 0.2]',
              show_default=False)
@click.option('--n_samples', '-n', default=None, type=int,
              help='The maximum number of samples to use in k-fold cross-validation from the non-testing portion of the dataset. If not specified, the whole non-testing portion is used.')
@click.option('--n_splits', default=5, help='The number k of splits for cross-validation.', show_default=True)
@click.option('--trn_size', default=0.75,
              help='The proportion of training data per fold in cross-validation. The remaining data of this fold is provided as evaluation data.',
              show_default=True)
@click.option('--batch_size', '-b', default=None, type=int,
              help='The batch size for training. If not specified, the full training set is used.')
@click.option('--n_epochs', '-e', default=1, help='The number of passes over the whole training set.',
              show_default=True)
@click.option('--max_hyperparam_sets', default=100,
              help='The maximum number of hyperparameter sets to try during hyperparameter optimization.',
              show_default=True)
@click.option('--max_layers', default=3, help='The maximum number of dense layers in the network when performing hyperparameter optimization.', show_default=True)
def main(alignments, spectrograms, operation, model_dir, tst_size, n_samples, n_splits, trn_size, batch_size, n_epochs,
         max_hyperparam_sets, max_layers):
    # Parse the test set size
    if tst_size == None:
        tst_size = 0.2
    else:
        error_message = 'Error: Invalid value for "--tst_size": "%s" is neither a valid integer nor a valid float' % (
            tst_size,)
        if '.' in tst_size or 'e' in tst_size or 'E' in tst_size:
            try:
                tst_size = float(tst_size)
            except ValueError:
                print(error_message)
        else:
            try:
                tst_size = int(tst_size)
            except ValueError:
                print(error_message)

    # Verify that the there was at least one operation requested
    if not (alignments or spectrograms or operation):
        print('No options given, try invoking the command with "--help" for help.')

    # Convert the data to a fastly loadable representation
    convert(alignments, spectrograms)

    # Create default model directory if there is none
    if model_dir is None:
        now = datetime.datetime.now()
        model_dir = os.path.dirname(os.path.abspath(__file__)) + ('/../models/%s_%s' % (now.date(), now.time()))
    model_dir = os.path.abspath(os.path.expanduser(model_dir))
    Path(model_dir).mkdir(exist_ok=True)
    print('Output directory: %s' % (model_dir,))

    # Setup logging
    # Change verbosity to info and log everything to a file (i.e. ../<model_dir>/main.log)
    tf.logging.set_verbosity(logging.INFO)
    logging.getLogger('tensorflow').handlers = [
        logging.FileHandler(model_dir + '/main.log'),
        # logging.StreamHandler(sys.stdout)
    ]
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress tensorflow debugging output
    print('Tensorflow debugging output is suppressed')

    # Handle data loading
    loader = DataLoader(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../dat/fast_load'),
                        tst_size=tst_size, seed=SEED)
    if operation == 'hyperoptimize':  # Hyperparameter optimization
        # Define hyperparameter search space
        @scope.define
        def to_int(number):
            return int(number)
        
        hyperparam_space = dict(
            lstm_size=scope.to_int(hp.qloguniform('lstm_size', 2.0, 5.0, 1)),
            dense_sizes=hp.choice('n_layers', [
                tuple(
                    scope.to_int(hp.qloguniform('dense_size_%d_%d' % (i, j), 2.0, 5.0, 1))
                    for j in range(i))
                for i in range(max_layers + 1)]),
            dropout=hp.uniform('dropout', 0.0, 1.0))

        # Optimize design via cross-validation
        trial_index = iter(count())

        def objective(hyperparams):
            trial_id = next(trial_index)
            progress.print_bar(trial_id, max_hyperparam_sets, 20, 'Trial:            ┃', '┃', '\n')
            trial_dir = model_dir + ('/trial_%d' % (trial_id,))
            if not os.path.exists(trial_dir):
                os.makedirs(trial_dir)
            with open(trial_dir + '/hyperparams.json', 'w') as hyperparams_file:
                json.dump(hyperparams, hyperparams_file)
            loss = cross_validate(trial_dir,
                                  loader,
                                  n_samples,
                                  n_splits,
                                  trn_size,
                                  batch_size,
                                  n_epochs,
                                  **hyperparams)
            report = {'loss': loss, 'status': STATUS_OK}
            with open(trial_dir + '/report.json', 'w') as report_file:
                json.dump(report, report_file)
            return report

        trials = Trials()
        hyperparams_best = fmin(fn=objective,
                                space=hyperparam_space,
                                algo=tpe.suggest,
                                max_evals=max_hyperparam_sets,
                                trials=trials)
        progress.print_bar(next(trial_index), max_hyperparam_sets, 20, 'Trial:            ┃', '┃', '\n')

        # Report results of hyperparameter optimization
        print('Best hyperparameters:')
        for param, value in sorted(hyperparams_best.items()):
            print('    %s: %s' % (param, value))
        with open(model_dir + '/hyperparams_best.json', 'w') as hyperparams_best_file:
            json.dump(hyperparams_best, hyperparams_best_file)
    # elif operation == 'train':
    #     train(estimator, loader, n_epochs)
    # elif operation == 'evaluate':
    #     evaluate(estimator, loader)
    elif operation == 'predict':
        predict(model_dir, loader, n_samples, n_splits, trn_size, batch_size, n_epochs, lstm_size) # TODO Add hyperparams


def convert(alignments, spectrograms):
    input_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../dat/speech_tokenizer')
    output_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../dat/fast_load')
    utterances_output_dir = output_dir + '/utterances'
    if not os.path.exists(utterances_output_dir):
        os.makedirs(utterances_output_dir)
    if alignments:
        print('Storing tag information...', end='', flush=True)
        tags = []
        with open(input_dir + '/new_alignments/phones.txt', 'r') as file_handle:
            for i, line in enumerate(file_handle):
                tag, id = line.strip('\n').split(' ')
                assert i == int(id)
                tag = tag.split('_')
                tag = (tag[1] if len(tag) > 1 else None, tag[0])
                tags.append(tag)
        with open(output_dir + '/tags.json', 'w') as file_handle:
            json.dump(tags, file_handle)
        print(' DONE')
        print('Converting alignments...', end='', flush=True)
        tag_dict = dict()
        with open(input_dir + '/new_alignments/merged_alignment.txt', 'r') as file_handle:
            for line in file_handle:
                # Get the data on one tag
                line = line.strip('\n').split(' ')
                # Convert starts and durations from seconds to numbers of frames without risking floating point errors
                assert len(re.sub('.*\.', '', line[2])) == 3
                assert len(re.sub('.*\.', '', line[3])) == 3
                id, start, duration, tag = line[0], int(re.sub('\.', '', line[2][:-1])), int(re.sub('\.', '', line[3][:-1])), int(line[4])
                if id not in tag_dict:
                    tag_dict[id] = []
                tag_dict[id].append((start, duration, tag))
        n_ids = len(tag_dict.keys())
        print(' DONE')
        start_time = time()
        for i, (key, value) in enumerate(tag_dict.items()):
            progress.print_bar(i, n_ids, 20, 'Storing alignment data... ┃', '┃')
            with open(utterances_output_dir + '/%s.json' % (key,), 'w') as file_handle:
                # file_dict = {'id' : key, 'alignments' : value}
                json.dump(value, file_handle)
        progress.print_bar(i + 1, n_ids, 20, 'Storing alignment data... ┃', '┃ DONE %.4fs' % (time() - start_time))
        print('Storing sequence IDs...', end='', flush=True)
        with open(output_dir + '/utterances.json', 'w') as file_handle:
            json.dump(sorted(tag_dict), file_handle, indent=4) # Asserts the same IDs in both alignments and spectrograms
        print(' DONE')
    if spectrograms:
        # Convert spectrograms
        print('Loading spectrogram data (this may take some time)...', end='', flush=True)
        feats, ids = readArk(input_dir + '/TEDLIUM_fbank_train_cleaned/unnormalized.feats.ark')
        n_ids = len(ids)
        print(' DONE')
        start_time = time()
        for i, (feat, id) in enumerate(zip(feats, ids)):
            progress.print_bar(i, n_ids, 20, 'Storing spectrogram data... ┃', '┃')
            np.save(utterances_output_dir + '/%s.npy' % (id,), feat)
        progress.print_bar(i + 1, n_ids, 20, 'Storing spectrogram data... ┃', '┃ DONE %.4fs' % (time() - start_time))


def cross_validate(model_dir, loader, n_samples, n_splits, trn_size, batch_size, n_epochs, lstm_size, dense_sizes, dropout):
    maximize_batch_size = batch_size is None
    loss = 0.0
    random_state = RandomState(BATCH_SEED)
    for i, (trn_ids, evl_ids, val_ids) in enumerate(loader.kfolds_ids(n_samples=n_samples, n_splits=n_splits, trn_size=trn_size)):
        progress.print_bar(i, n_splits, 20, '    Fold:         ┃', '┃', '\n')

        # Build estimator
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            params=dict(lstm_size=lstm_size,
                        dense_sizes=dense_sizes,
                        dropout=dropout),
            model_dir=model_dir + '/fold_%d' % (i,)
        )

        # Initialize weights:
        # Train on nothing to make sure that a zero-global-step checkpoint exists.
        # Any evaluation prior to this may be incorrect due to a different random initialization.
        estimator.train(input_fn=lambda: input_fn(loader, [], 1, random_state, mode=TRAIN_MODE), steps=None)

        # Train/evaluate for multiple epochs
        eval_result = estimator.evaluate(input_fn=lambda: input_fn(loader, evl_ids, mode=EVAL_MODE))  # for tensorboard correct display ..
        batch_size = len(trn_ids) if maximize_batch_size else batch_size
        for epoch in range(n_epochs):
            progress.print_bar(epoch, n_epochs, 20, '        Epoch:    ┃', '┃')
            estimator.train(input_fn=lambda: input_fn(loader, trn_ids, batch_size, random_state, mode=TRAIN_MODE), steps=None)  # Training
            eval_result = estimator.evaluate(input_fn=lambda: input_fn(loader, evl_ids, mode=EVAL_MODE))  # for possible early stopping
        progress.print_bar(epoch + 1, n_epochs, 20, '        Epoch:    ┃', '┃')

        # Validate
        eval_result = estimator.evaluate(input_fn=lambda: input_fn(loader, val_ids, mode=EVAL_MODE))  # Validation TODO Is logged as evalutation
        print('        Accuracy:  %.6f' % (eval_result['acc'],))
        print('        Precision: %.6f' % (eval_result['precision'],))
        print('        Recall:    %.6f' % (eval_result['recall'],))
        print('        F1:        %.6f' % (eval_result['f1'],))
        loss += eval_result['loss']

    # Report hyperparameter set performance
    loss /= n_splits
    progress.print_bar(i + 1, n_splits, 20, '    Fold:         ┃', '┃\n    Loss:          %f' % (loss,), '\n')
    return loss


def train(estimator, loader, n_epochs):  # TODO
    raise NotImplementedError


def evaluate(estimator, loader):  # TODO
    raise NotImplementedError
    # eval_result = estimator.evaluate(input_fn=eval_input_fn)
    # print(eval_result)


# TODO
def predict(model_dir, loader, n_samples, n_splits, trn_size, batch_size, n_epochs, lstm_size):
    # raise NotImplementedError
    params = {
        'dropout': 0.5,
        'lstm_size': lstm_size,
        'learning_rate': 0.1
    }
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        model_dir=model_dir
    )

    predictions = estimator.predict(input_fn=lambda: input_fn(loader, loader.ids[:1], mode=PRED_MODE))
    for pred_dict in predictions:
        class_id_seq = list(pred_dict['predictions'].reshape(-1))
        prob_seq = list(pred_dict['probabilities'].reshape(-1))
        feat_seq, align_seq, _ = loader.load(loader.ids[0])
        label_seq = list(align_seqs_to_breaking_labels(align_seq, feat_seq.shape[0]))
        print(len(class_id_seq),len(prob_seq), len(label_seq), len(feat_seq))
        print('Preds:  %s' % (class_id_seq,), 'Labels: %s' % (label_seq,), sep='\n')
        print('Probs:  %s' % (prob_seq,), 'Labels: %s' % (label_seq,), sep='\n')


if __name__ == '__main__':
    main()

    # plt.switch_backend('QT4Agg')
    # loader = DataLoader(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../dat/fast_load'), tst_size=20, seed=SEED)
    # print('Test set IDs: %s' % (loader.test_set_ids(),))
    # feat_seqs, align_seqs, phone_seqs = loader.load(['spk0000000_EvaZeisel_2001-0002861-0003242-1', 'spk0001415_ArthurPottsDawson_2010G-0001676-0002585-1'])
    # print(align_seqs_to_breaking_labels(align_seqs, [feat_seq.shape[0] for feat_seq in feat_seqs]))
    # loader.plot(['spk0000000_EvaZeisel_2001-0002861-0003242-1', 'spk0001415_ArthurPottsDawson_2010G-0001676-0002585-1'])
    # loader.plot('spk0000000_EvaZeisel_2001-0025391-0026099-1')
    # for i, ((trn_ids, trn_feat_seqs, trn_align_seqs, trn_phone_seqs),
    #         (evl_ids, evl_feat_seqs, evl_align_seqs, evl_phone_seqs),
    #         (val_ids, val_feat_seqs, val_align_seqs, val_phone_seqs)) in enumerate(loader.kfolds(n_samples=50, n_splits=5, trn_size=0.75)):
    #     print('Fold %d' % (i,))
    #     print(trn_ids, trn_feat_seqs, trn_align_seqs, trn_phone_seqs, sep='\n')
    #     print(evl_ids, evl_feat_seqs, evl_align_seqs, evl_phone_seqs, sep='\n')
    #     print(val_ids, val_feat_seqs, val_align_seqs, val_phone_seqs, sep='\n')
    #     print()

# main.py ends here
