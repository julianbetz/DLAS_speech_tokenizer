#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:47:15 2017

@author: Benjamin Milde (Language Technology, Universität Hamburg, Germany)
"""

from __future__ import division
from __future__ import unicode_literals  
from builtins import bytes, chr, int, str
#
# This code seems to be adapted from https://github.com/MaigoAkisame/fileutils/blob/master/kaldi.py
# The original code was in Python 2.7, it has been adadpted/changed so that it is compatible with Python 3.x
# Warnings: no effort has been made to test backwards compatiblity, the imports above may or may not make this work with Python 2.x, too
#

import numpy
import struct
from utils import smart_open
import io

try:
    import resource
    filelimit=200000
    resource.setrlimit(resource.RLIMIT_NOFILE, (filelimit,filelimit))
except:
    print('Kaldi_io: tried increasing the open file limit to', filelimit, 'but it didnt work. If you have many matrices and want mmap support for kaldi io, consider increasing the global system limit for open files.')

def readString(f):
    s = b""
    while True:
        c = f.read(1)
        if c == b"": raise ValueError("EOF encountered while reading a string.")
        if c == b" ": return s
        s += c

def readInteger(f):
    n = ord(f.read(1))
    integer = int.from_bytes(f.read(n), byteorder='little') # read an int from binary, python3 only
    return integer
    # python2 code (the above will not work in python2)
    #return reduce(lambda x, y: x * 256 + ord(y), f.read(n)[::-1], 0)

def readMatrix(f):
    header = f.read(2)
    if header != b"\0B":
        raise ValueError("Binary mode header ('\0B') not found when attempting to read a matrix.")
    format = readString(f)
    nRows = readInteger(f)
    nCols = readInteger(f)

    if format == b"DM":
        data = struct.unpack("<%dd" % (nRows * nCols), f.read(nRows * nCols * 8))
        data = numpy.array(data, dtype = "float64")
    elif format == b"FM":
        data = struct.unpack("<%df" % (nRows * nCols), f.read(nRows * nCols * 4))
        data = numpy.array(data, dtype = "float32")
    else:
        raise ValueError("Unknown matrix format '%s' encountered while reading; currently supported formats are DM (float64) and FM (float32)." % format)
    return data.reshape(nRows, nCols)

def writeString(f, s):
    f.write(s + b" ")

def writeInteger(f, a):
    s = struct.pack(b"<i", a)
    f.write(chr(len(s)).encode('latin-1') + s)

def writeMatrix(f, data):
    f.write(b"\0B")      # Binary data header
    if str(data.dtype) == "float64":
        writeString(f, b"DM")
        writeInteger(f, data.shape[0])
        writeInteger(f, data.shape[1])
        f.write(struct.pack(b"<%dd" % data.size, *data.ravel()))
    elif str(data.dtype) == "float32":
        writeString(f, b"FM")
        writeInteger(f, data.shape[0])
        writeInteger(f, data.shape[1])
        f.write(struct.pack(b"<%df" % data.size, *data.ravel()))
    else:
        raise ValueError("Unsupported matrix format '%s' for writing; currently supported formats are float64 and float32." % str(data.dtype))

def readMemmapCache(memmap_dir='', memmap_dtype='float32'):
    features = []
    uttids = []
    with io.open(memmap_dir + '/' + 'feature_map', 'r') as feature_map:
        for line in feature_map:
            if line[-1] == '\n':
                line = line[:-1]
            split = line.split()
            uttid = split[0]
            featshape = (int(split[1]), int(split[2]))
            feature_mmap = numpy.memmap(memmap_dir + '/' + uttid, dtype=memmap_dtype, mode='r', shape=featshape)
            
            uttids.append(uttid)
            features.append(feature_mmap)
    return features, uttids

def readArk(filename, limit = numpy.inf, memmap_dir='', memmap_dtype='float32'):
    """
    Reads the features in a Kaldi ark file.
    Returns a list of feature matrices and a list of the utterance IDs.
    """
    features = []
    uttids = []
    with smart_open(filename, "rb") as f:
        while True:
            try:
                uttid = readString(f).decode('utf-8')
            except ValueError:
                break
            feature = readMatrix(f)
            # use a memmap dir to hold the array content on a disk (e.g. ssd cache that is larger than your main memory)
            if memmap_dir!='':
                feature_mmap = numpy.memmap(memmap_dir + '/' + uttid, dtype=memmap_dtype, mode='w+', shape=feature.shape)
                feature_mmap[:] = feature[:]
                feature_mmap.flush()
                features.append(feature_mmap)
                del feature
            else:
                features.append(feature)
            uttids.append(uttid)
            if len(features) == limit: break
    if memmap_dir!='':
        with io.open(memmap_dir + '/' + 'feature_map', 'w') as feature_map:
            for uttid, feature in zip(uttids, features):
                feature_map.write(uttid + " %i %i\n" % (feature.shape[0], feature.shape[1]))
    return features, uttids

def readScp(filename, limit = numpy.inf, memmap_dir='', memmap_dtype='float32'):
    """
    Reads the features in a Kaldi script file.
    Returns a list of feature matrices and a list of the utterance IDs.
    """
    features = []
    uttids = []
    with smart_open(filename, "r") as f:
        for line in f:
            uttid, pointer = line.strip().split()
            p = pointer.rfind(":")
            arkfile, offset = pointer[:p], int(pointer[p+1:])
            with smart_open(arkfile, "rb") as g:
                g.seek(offset)
                feature = readMatrix(g)
            features.append(feature)
            uttids.append(uttid)
            if len(features) == limit: break
    return features, uttids

def writeArk(filename, features, uttids, append=False):
    """
    Takes a list of feature matrices and a list of utterance IDs,
      and writes them to a Kaldi ark file.
    Returns a list of strings in the format "filename:offset",
      which can be used to write a Kaldi script file.
    """
    pointers = []
    with smart_open(filename, "ab" if append else "wb") as f:
        for feature, uttid in zip(features, uttids):
            writeString(f, uttid.encode('utf-8'))
            pointers.append("%s:%d" % (filename, f.tell()))
            writeMatrix(f, feature)
    return pointers

def writeScp(filename, uttids, pointers, append=False):
    """
    Takes a list of utterance IDs and a list of strings in the format "filename:offset",
      and writes them to a Kaldi script file.
    """
    with smart_open(filename, "a" if append else "w") as f:
        for uttid, pointer in zip(uttids, pointers):
            f.write("%s %s\n" % (uttid, pointer))
