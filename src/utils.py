#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Benjamin Milde
"""

license = '''

Copyright 2017,2018 Benjamin Milde (Language Technology, Universität Hamburg, Germany)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

import wave
import numpy as np
import scipy
import os
import scipy.io.wavfile
import tensorflow as tf
import os.path
import gzip
import bz2
import wavefile
from collections import defaultdict

def smart_open(filename, mode = 'rb', *args, **kwargs):
    '''
    Opens a file "smartly":
      * If the filename has a ".gz" or ".bz2" extension, compression is handled
        automatically;
      * If the file is to be read and does not exist, corresponding files with
        a ".gz" or ".bz2" extension will be attempted.
    '''
    readers = {'.gz': gzip.GzipFile, '.bz2': bz2.BZ2File}
    if 'r' in mode and not os.path.exists(filename):
        for ext in readers:
            if os.path.exists(filename + ext):
                filename += ext
                break
    extension = os.path.splitext(filename)[1]
    return readers.get(extension, open)(filename, mode, *args, **kwargs)

#compresses the dynamic range, see https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
def encode_mulaw(signal,mu=255):
    return np.sign(signal)*(np.log1p(mu*np.abs(signal)) / np.log1p(mu))

#uncompress the dynamic range, see https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
def decode_mulaw(signal,mu=255):
    return np.sign(signal)*(1.0/mu)*(np.power(1.0+mu,np.abs(signal))-1.0)

# discretize signal between -1.0 and 1.0 into mu+1 bands.
def discretize(signal, mu=255.0):
    output = np.array(signal)
    output += 1.0
    output = output*(0.5*mu)
    signal = np.fmax(0.0,output)
    #signal = np.fmin(255.0,signal)
    return signal.astype(np.int32)

def undiscretize(signal, mu=255.0):
    output = np.array(signal)
    output = output.astype(np.float32)
    output /= 0.5*mu
    output -= 1.0
    signal = np.fmax(-1.0,output)
    signal = np.fmin(1.0,signal)
    return signal

def readWordPosFile(filename,pos1=0,pos2=1):
    unalign_list = []
    with open(filename) as f:
        for line in f.readlines():
            split = line[:-1].split(" ")
            unalign_list.append((float(split[pos1]), float(split[pos2])))
    return unalign_list

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def loadIdFile(idfile,use_no_files=-1):
    ids = []
    with open(idfile) as f:
        ids = f.read().split('\n')[:use_no_files]
    
    ids = [myid for myid in ids if myid != '']
    
    if len(ids[0].split()) > 1:
        utt_ids = []
        wav_files = []
        
        for myid in ids:
            print(myid)
            split = myid.split()
            utt_ids.append(split[0])
            wav_files.append(split[1])
    else:
        utt_ids = []
        wav_files = ids
    #check if ids exist
    #ids = [myid for myid in ids if os.path.ispath(myid)]
    return utt_ids, wav_files

def loadPhnFile(phn_file):
    positions = []
    names = []
    with open(phn_file) as phn:
        for line in phn:
            if line[-1] == '\n':
                line = line[:-1]
            split = line.split()
            pos = (split[0],split[1])
            name = split[-1]
            positions.append(pos)
            names.append(name)
    return positions,names

def loadUtt2Spk(utt_filename):
    utts = {}
    with open(utt_filename) as utt_file:
        for line in utt_file:
            if line[-1] == '\n':
                line = line[:-1]
            split = line.split()
            utt = split[0]
            spk = split[1]
            utts[utt] = spk
    return utts

def loadSpk2Utt(utt_filename, ignore_dash_spk_id=True):
    spks = defaultdict(list)
    with open(utt_filename) as utt_file:
        for line in utt_file:
            if line[-1] == '\n':
                line = line[:-1]
            split = line.split()
            spk = split[0]
            if ignore_dash_spk_id and '-' in spk:
                spk = spk.split('-')[0]
            utt = split[1:]
            spks[spk] += utt
    return spks

def getSignalOld(utterance):
    spf = wave.open(utterance, 'r')
    sound_info = spf.readframes(-1)
    signal = np.fromstring(sound_info, 'Int16')
    return signal, spf.getframerate()

# This is needed since the old loader had problems with NIST headers from TIMIT. 
# See also https://stackoverflow.com/questions/10187043/read-nist-wav-file-in-timit-database-into-python-numpy-array
def getSignal(utterance):
    samplerate, signal = wavefile.load(utterance)
    print(signal)
    signal = signal[0]
    #print(utterance, 'dtype:', signal.dtype, 'min:', min(signal), 'max:', max(signal), 'samplerate:', samplerate)
    return signal, samplerate

def writeSignal(signal, myfile, rate=16000, do_decode_mulaw=False):
    if do_decode_mulaw:
        signal = decode_mulaw(signal)
    return scipy.io.wavfile.write(myfile, rate, signal)

def rolling_window(a, window_len, hop):
    print("a.shape[:-1]", a.shape[:-1])
    print("a.shape[-1]", a.shape[-1])
    shape = a.shape[:-1] + (a.shape[-1] - window_len + 1, window_len)
    strides = a.strides + (a.strides[-1],)
    print('shape:',shape)
    print('strides:',strides)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::hop]

# This code is from https://gist.github.com/seberg/3866040, public domain?
# This function is not licensed under Apache 2.0
def rolling_window_better(array, window=(0,), asteps=None, wsteps=None, axes=None, toend=True):
    """Create a view of `array` which for every point gives the n-dimensional
    neighbourhood of size window. New dimensions are added at the end of
    `array` or after the corresponding original dimension.
    
    Parameters
    ----------
    array : array_like
        Array to which the rolling window is applied.
    window : int or tuple
        Either a single integer to create a window of only the last axis or a
        tuple to create it for the last len(window) axes. 0 can be used as a
        to ignore a dimension in the window.
    asteps : tuple
        Aligned at the last axis, new steps for the original array, ie. for
        creation of non-overlapping windows. (Equivalent to slicing result)
    wsteps : int or tuple (same size as window)
        steps for the added window dimensions. These can be 0 to repeat values
        along the axis.
    axes: int or tuple
        If given, must have the same size as window. In this case window is
        interpreted as the size in the dimension given by axes. IE. a window
        of (2, 1) is equivalent to window=2 and axis=-2.       
    toend : bool
        If False, the new dimensions are right after the corresponding original
        dimension, instead of at the end of the array. Adding the new axes at the
        end makes it easier to get the neighborhood, however toend=False will give
        a more intuitive result if you view the whole array.
    
    Returns
    -------
    A view on `array` which is smaller to fit the windows and has windows added
    dimensions (0s not counting), ie. every point of `array` is an array of size
    window.
    
    Examples
    --------
    >>> a = np.arange(9).reshape(3,3)
    >>> rolling_window(a, (2,2))
    array([[[[0, 1],
             [3, 4]],

            [[1, 2],
             [4, 5]]],


           [[[3, 4],
             [6, 7]],

            [[4, 5],
             [7, 8]]]])
    
    Or to create non-overlapping windows, but only along the first dimension:
    >>> rolling_window(a, (2,0), asteps=(2,1))
    array([[[0, 3],
            [1, 4],
            [2, 5]]])
    
    Note that the 0 is discared, so that the output dimension is 3:
    >>> rolling_window(a, (2,0), asteps=(2,1)).shape
    (1, 3, 2)
    
    This is useful for example to calculate the maximum in all (overlapping)
    2x2 submatrixes:
    >>> rolling_window(a, (2,2)).max((2,3))
    array([[4, 5],
           [7, 8]])
           
    Or delay embedding (3D embedding with delay 2):
    >>> x = np.arange(10)
    >>> rolling_window(x, 3, wsteps=2)
    array([[0, 2, 4],
           [1, 3, 5],
           [2, 4, 6],
           [3, 5, 7],
           [4, 6, 8],
           [5, 7, 9]])
    """
    array = np.asarray(array)
    orig_shape = np.asarray(array.shape)
    window = np.atleast_1d(window).astype(int) # maybe crude to cast to int...
    
    if axes is not None:
        axes = np.atleast_1d(axes)
        w = np.zeros(array.ndim, dtype=int)
        for axis, size in zip(axes, window):
            w[axis] = size
        window = w
    
    # Check if window is legal:
    if window.ndim > 1:
        raise ValueError("`window` must be one-dimensional.")
    if np.any(window < 0):
        raise ValueError("All elements of `window` must be larger then 1.")
    if len(array.shape) < len(window):
        raise ValueError("`window` length must be less or equal `array` dimension.") 

    _asteps = np.ones_like(orig_shape)
    if asteps is not None:
        asteps = np.atleast_1d(asteps)
        if asteps.ndim != 1:
            raise ValueError("`asteps` must be either a scalar or one dimensional.")
        if len(asteps) > array.ndim:
            raise ValueError("`asteps` cannot be longer then the `array` dimension.")
        # does not enforce alignment, so that steps can be same as window too.
        _asteps[-len(asteps):] = asteps
        
        if np.any(asteps < 1):
             raise ValueError("All elements of `asteps` must be larger then 1.")
    asteps = _asteps
    
    _wsteps = np.ones_like(window)
    if wsteps is not None:
        wsteps = np.atleast_1d(wsteps)
        if wsteps.shape != window.shape:
            raise ValueError("`wsteps` must have the same shape as `window`.")
        if np.any(wsteps < 0):
             raise ValueError("All elements of `wsteps` must be larger then 0.")

        _wsteps[:] = wsteps
        _wsteps[window == 0] = 1 # make sure that steps are 1 for non-existing dims.
    wsteps = _wsteps

    # Check that the window would not be larger then the original:
    if np.any(orig_shape[-len(window):] < window * wsteps):
        raise ValueError("`window` * `wsteps` larger then `array` in at least one dimension.")

    new_shape = orig_shape # just renaming...
    
    # For calculating the new shape 0s must act like 1s:
    _window = window.copy()
    _window[_window==0] = 1
    
    new_shape[-len(window):] += wsteps - _window * wsteps
    new_shape = (new_shape + asteps - 1) // asteps
    # make sure the new_shape is at least 1 in any "old" dimension (ie. steps
    # is (too) large, but we do not care.
    new_shape[new_shape < 1] = 1
    shape = new_shape
    
    strides = np.asarray(array.strides)
    strides *= asteps
    new_strides = array.strides[-len(window):] * wsteps
    
    # The full new shape and strides:
    if toend:
        new_shape = np.concatenate((shape, window))
        new_strides = np.concatenate((strides, new_strides))
    else:
        _ = np.zeros_like(shape)
        _[-len(window):] = window
        _window = _.copy()
        _[-len(window):] = new_strides
        _new_strides = _
        
        new_shape = np.zeros(len(shape)*2, dtype=int)
        new_strides = np.zeros(len(shape)*2, dtype=int)
        
        new_shape[::2] = shape
        new_strides[::2] = strides
        new_shape[1::2] = _window
        new_strides[1::2] = _new_strides
    
    new_strides = new_strides[new_shape != 0]
    new_shape = new_shape[new_shape != 0]
    
    return np.lib.stride_tricks.as_strided(array, shape=new_shape, strides=new_strides)


def writeArkTextFeatFile(feat, feat_name, out_filename, append = False):
    with open(out_filename, 'a' if append else 'w') as out_file:
        out_file.write(feat_name  + ' [')
        for feat_vec in feat:
            feat_vec_str = ' '.join([str(elem) for elem in feat_vec])
            out_file.write(feat_vec_str)

def writeZeroSpeechFeatFile(feat, out_filename, window_length, hop_size):
    ensure_dir(out_filename)
    with open(out_filename, 'w') as out_file:
        for i,feat_vec in enumerate(feat):
            pos = i * hop_size + (window_length / 2.0)
            feat_vec_str = ' '.join([str(elem) for elem in feat_vec])
            out_file.write(str(pos) + ' ' + feat_vec_str + '\n')
            
def tensor_normalize_0_to_1(in_tensor):
    x_min = tf.reduce_min(in_tensor)
    x_max = tf.reduce_max(in_tensor)
    tensor_0_to_1 = ((in_tensor - x_min) / (x_max - x_min))
    return tensor_0_to_1
