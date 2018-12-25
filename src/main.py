#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Filename: main.py
# Author: Julian Betz
# Created: 2018-12-20
# Version: 2018-12-25
#
# Description:
#     Entry point to the training/testing procedures.

import os
import click
import re
from time import time
import numpy as np
from matplotlib import pyplot as plt
from kaldi_io import readArk
import json

from data_handler import DataLoader
from util import progress

@click.command()
# @click.option('--gpu/--cpu', '-g/-c', default=False, help='Whether to use a GPU.', show_default=True)
# @click.option('--n_samples', '-n', required=True, type=int, help='The number of samples to use, if available.')
@click.option('--alignments/--no-alignments', '-a/', default=False, help='Whether to convert alignment data', show_default=True)
@click.option('--spectrograms/--no-spectrograms', '-s/', default=False, help='Whether to convert the spectrograms', show_default=True)
def main(alignments, spectrograms):
    if not (alignments or spectrograms):
        print('No options given, try invoking the command with "--help" for help.')
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

if __name__ == '__main__':
    main()

    # plt.switch_backend('QT4Agg')
    # loader = DataLoader(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../dat/fast_load'), tst_size=20, seed=1000)
    # # print(loader.load(['spk0000000_EvaZeisel_2001-0002861-0003242-1', 'spk0001415_ArthurPottsDawson_2010G-0001676-0002585-1']))
    # loader.plot(['spk0000000_EvaZeisel_2001-0002861-0003242-1', 'spk0001415_ArthurPottsDawson_2010G-0001676-0002585-1'])
    # loader.plot('spk0000000_EvaZeisel_2001-0002861-0003242-1')
    # for i, ((trn_ids, trn_feat_seqs, trn_align_seqs, trn_phone_seqs),
    #         (evl_ids, evl_feat_seqs, evl_align_seqs, evl_phone_seqs),
    #         (val_ids, val_feat_seqs, val_align_seqs, val_phone_seqs)) in enumerate(loader.kfolds(n_samples=10, n_splits=5, trn_size=0.625)):
    #     print('Fold %d' % (i,))
    #     print(trn_ids, trn_feat_seqs, trn_align_seqs, trn_phone_seqs, sep='\n')
    #     print(evl_ids, evl_feat_seqs, evl_align_seqs, evl_phone_seqs, sep='\n')
    #     print(val_ids, val_feat_seqs, val_align_seqs, val_phone_seqs, sep='\n')
    #     print()
    
# main.py ends here
