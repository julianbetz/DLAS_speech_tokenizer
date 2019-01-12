# -*- coding: utf-8 -*-

# Filename: data_handler.py
# Author: Julian Betz
# Created: 2018-12-23
# Version: 2019-01-12
# 
# Description: A class for converting and loading the dataset.

import sys
import os
import numpy as np
from numpy.random import RandomState
from tensorflow.data import Dataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import json

class DataLoader:
    TAGS_ALL = [None, 'B', 'I', 'E', 'S']
    TAG_COLORS = [(0, 0, 0, 0), (0, 1, 0, 0.2), (0, 0, 1, 0.2), (1, 0, 0, 0.2), (1, 1, 0, 0.2)]
    
    def __init__(self, data_dir, tst_size, seed=None):
        """A data management utility.
        
        Args:
            data_dir (str): The directory where the dataset is stored.
            tst_size (int/float): The number/proportion of samples from the complete dataset to reserve for testing.
            seed (int/None): The seed used for shuffling the dataset. If None, a random seed is generated.
        """
        self.data_dir = os.path.abspath(data_dir)
        with open(self.data_dir + '/tags.json', 'r') as file_handle:
            self.tags = json.load(file_handle)
        with open(self.data_dir + '/utterances.json', 'r') as file_handle:
            self.ids = json.load(file_handle)
        self.tst_size = tst_size if type(tst_size) == int else int(tst_size * len(self.ids))
        assert self.tst_size >= 0 and self.tst_size <= len(self.ids)
        self.seed = RandomState().randint((2 ** 31 - 1) + 2 ** 31) if seed is None else seed

    def load(self, ids):
        """Loads the data for the specified IDs from disk and returns it.

        A single ID by itself can be provided as a string.

        Args:
            ids (str/iterable<str>): The ID / the collection of IDs that should be loaded. Data is returned in the order in which the IDs are provided.
        Returns:
            feat_seqs (numpy.ndarray/list<numpy.ndarray>): The features for the specified IDs.
            align_seqs (numpy.ndarray/list<numpy.ndarray>): The alignment data for the specified IDs.
            phone_seqs (list<str>/list<list<str>>): Phonetic data for the specified IDs.
        """
        tags_all_inv = {v : i for i, v in enumerate(DataLoader.TAGS_ALL)}
        if type(ids) == str:
            feat_seq = np.load(self.data_dir + '/utterances/%s.npy' % (ids,))
            with open(self.data_dir + '/utterances/%s.json' % (ids,), 'r') as file_handler:
                align_seq = np.array(json.load(file_handler), dtype=np.uint32)
            phone_seq = []
            for token in align_seq:
                tag = self.tags[token[2]]
                token[2] = tags_all_inv[tag[0]]
                phone_seq.append(tag[1])
            return feat_seq, align_seq, phone_seq
        else:
            feat_seqs = [np.load(self.data_dir + '/utterances/%s.npy' % (i,)) for i in ids]
            align_seqs = []
            phone_seqs = []
            for i in ids:
                with open(self.data_dir + '/utterances/%s.json' % (i,), 'r') as file_handler:
                    align_seqs.append(np.array(json.load(file_handler), dtype=np.uint32))
                phone_seqs.append([])
                for token in align_seqs[-1]:
                    tag = self.tags[token[2]]
                    token[2] = tags_all_inv[tag[0]]
                    phone_seqs[-1].append(tag[1])
            return feat_seqs, align_seqs, phone_seqs

    def plot(self, ids):
        """Displays the data for the specified IDs as a plot.

        A single ID by itself can be provided as a string.

        Args:
            ids (str/iterable<str>): The ID / the collection of IDs that should be displayed. Data is plotted in the order in which the IDs are provided.
        """
        for i, feat_seq, align_seq, phone_seq in (((ids, *self.load(ids)),) if type(ids) == str else zip(ids, *self.load(ids))):
            plt.imshow(feat_seq.transpose(), origin='lower', cmap='gray_r')
            for token in align_seq:
                rect = Rectangle((token[0], 0), token[1], feat_seq.shape[1], linewidth=0, facecolor=DataLoader.TAG_COLORS[token[2]])
                # plt.axvline(token[0], c=['black', 'green', 'blue', 'red', 'yellow'][token[2]])
                plt.gca().add_patch(rect)
            plt.xticks(align_seq[:, 0], phone_seq)
            plt.yticks([], [])
            plt.title(i)
            plt.tight_layout()
            plt.get_current_fig_manager().window.showMaximized()
            plt.show()

    def test_set_ids(self):
        """Returns the IDs for the test set.
        """
        r = RandomState(self.seed)
        ids = shuffle(self.ids, random_state=r, n_samples=self.tst_size)
        print(ids, type(ids))
        return ids

    def test_set(self):
        tst_ids = self.test_set_ids()
        return (tst_ids, *self.load(tst_ids))

    def kfolds_ids(self, n_samples=None, n_splits=10, trn_size=0.8):
        """Iterates over folds to provide the IDs with which to index the appropriate data.

        Shuffles the data IDs using a random number generator seeded with this class's seed, splits off the testing portion and provides training, evaluation and validation data IDs per fold from a portion of the remaining data.
        
        Args:
            n_samples (int/None): The maximum number of samples to use in k-fold cross-validation from the non-testing portion of the dataset. If None, the whole non-testing portion is used.
            n_splits (int): The number k of splits for cross-validation.
            trn_size (float): The proportion of training data per fold. The remaining data of this fold is provided as evaluation data.
        Yields:
            trn_ids (list<str>): A list of sequence IDs for training.
            evl_ids (list<str>): A list of sequence IDs for evaluation.
            val_ids (list<str>): A list of sequence IDs for validation.
        Example:
            for i, (trn_ids, evl_ids, val_ids) in enumerate(dl.kfolds()):
                trn_data, evl_data, val_data = dl.load(trn_ids), dl.load(evl_ids), dl.load(val_ids)
                pass # Use data for training/evaluation/validation on fold i
        """
        r = RandomState(self.seed)
        n_samples = min(float('inf') if n_samples is None else n_samples, len(self.ids) - self.tst_size)
        ids = shuffle(self.ids, random_state=r, n_samples=self.tst_size+n_samples)[self.tst_size:]
        kfolder = KFold(n_splits=n_splits, shuffle=False)
        for dev_indices, val_indices in kfolder.split(ids):
            trn_indices, evl_indices = train_test_split(dev_indices, test_size=None, train_size=trn_size, shuffle=True, random_state=r)
            yield ([ids[trn_index] for trn_index in trn_indices],
                   [ids[evl_index] for evl_index in evl_indices],
                   [ids[val_index] for val_index in val_indices])

    def kfolds(self, n_samples=None, n_splits=10, trn_size=0.8):
        """Iterates over folds to provide the appropriate data.

        Shuffles the data using a random number generator seeded with this class's seed, splits off the testing portion and provides training, evaluation and validation data per fold from a portion of the remaining data.
        Only use this method if the fold data is not expected to take up large amounts of memory.

        Args:
            n_samples (int/None): The maximum number of samples to use in k-fold cross-validation from the non-testing portion of the dataset. If None, the whole non-testing portion is used.
            n_splits (int): The number k of splits for cross-validation.
            trn_size (float): The proportion of training data per fold. The remaining data of this fold is provided as evaluation data.
        Yields:
            trn_data (tuple<list>): Data for training, consisting of:
                trn_ids (list<str>): A list of sequence IDs for training.
                trn_feat_seqs (list<numpy.ndarray>): A list of feature sequences for training.
                trn_align_seqs (list<numpy.ndarray>): A list of alignment data for training.
                trn_phone_seqs (list<list<str>>): A list of phone sequences for training.
            evl_data (tuple<list>): Data for evaluation, consisting of:
                evl_ids (list<str>): A list of sequence IDs for evaluation.
                evl_feat_seqs (list<numpy.ndarray>): A list of feature sequences for evaluation.
                evl_align_seqs (list<numpy.ndarray>): A list of alignment data for evaluation.
                evl_phone_seqs (list<list<str>>): A list of phone sequences for evaluation.
            val_data (tuple<list>): Data for validation, consisting of:
                val_ids (list<str>): A list of sequence IDs for validation.
                val_feat_seqs (list<numpy.ndarray>): A list of feature sequences for validation.
                val_align_seqs (list<numpy.ndarray>): A list of alignment data for validation.
                val_phone_seqs (list<list<str>>): A list of phone sequences for validation.
        Example:
            for i, ((ti, tf, ta, tp), (ei, ef, ea, ep), (vi, vf, va, vp)) in enumerate(dl.kfolds()):
                pass # Use data for training/evaluation/validation on fold i
        """
        for trn_ids, evl_ids, val_ids in self.kfolds_ids(n_samples=n_samples, n_splits=n_splits, trn_size=trn_size):
            yield ((trn_ids, *self.load(trn_ids)),
                   (evl_ids, *self.load(evl_ids)),
                   (val_ids, *self.load(val_ids)))

def align_seqs_to_alternating_labels(align_seqs, lengths):
    """Converts the specified alignment data to labels.

    Labels encode word boundaries in checkerboard representation.
    A single align sequence by itself can be provided as a list of rank 2.

    Args:
        align_seqs (numpy.ndarray/list<numpy.ndarray>): The alignment data.
        lengths (int/list<int>): The total number of frames for the alignment sequences.
    Returns:
        feat_seqs (numpy.ndarray/list<numpy.ndarray>): The labels corresponding to the specified alignment data.
    """
    if type(align_seqs[0][0]) not in [list, tuple, np.ndarray]: # Single alignment sequence
        label_seq = np.full((lengths,), 0.5, dtype=np.float32)
        current_class = 0.0
        for i, (start, duration, tag) in enumerate(align_seqs):
            label_seq[start:start+duration] = current_class
            if (DataLoader.TAGS_ALL[tag] == 'E'
                or DataLoader.TAGS_ALL[tag] == 'S'
                or (DataLoader.TAGS_ALL[tag] == None
                    and align_seqs.shape[0] > i + 1
                    and DataLoader.TAGS_ALL[align_seqs[i + 1][2]] != None)): # Assumes that all non-phone tags can be treated as the same token
                current_class = 1.0 - current_class
        return label_seq
    else:                               # List of alignment sequences
        label_seqs = [align_seqs_to_alternating_labels(align_seq, length) for align_seq, length in zip(align_seqs, lengths)]
        return label_seqs

def train_input_fn(features, labels, batch_size): # TODO Remove batch_size
    dataset = Dataset.from_tensor_slices(({'features' : features}, labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size) # TODO
    return dataset.make_one_shot_iterator().get_next()         # TODO ???

def eval_input_fn(features, labels, batch_size): # TODO Remove batch_size
    dataset = Dataset.from_tensor_slices(({'features' : features}, labels))
    dataset = dataset.batch(batch_size)
    return dataset                      # TODO One shot-iterator necessary?
    
# data_handler.py ends here
