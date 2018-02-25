#!/usr/bin/env python
#coding=utf-8
'''
This module returns for data reader, used for model train or test or eval
contain:
    1.tf Data high level API reader, now suport:
    1.1 libsvm csv style
    1.2 deep wide csv style
    2.use queue read from disk
'''

import shutil
import os
import json
import glob
from datetime import date, timedelta
from time import time
import random
import pandas as pd
import numpy as np
import tensorflow as tf

# https://www.tensorflow.org/programmers_guide/datasets
def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
    print('Parsing', filenames)
    def decode_libsvm(line):
        columns = tf.string_split([line], ' ')
        labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
        splits = tf.string_split(columns.values[1:], ':')
        id_vals = tf.reshape(splits.values,splits.dense_shape)
        feat_ids, feat_vals = tf.split(id_vals,num_or_size_splits=2,axis=1)
        feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
        feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
        return {"feat_ids": feat_ids, "feat_vals": feat_vals} 

    dataset = tf.data.TextLineDataset(filenames).map(decode_libsvm, num_parallel_calls=10).prefetch(500000)  
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(num_epochs).batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

def input_csv_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False, csv_column, csv_column_default):
    def parse_csv(line):
        print('Parsing', filenames)
        columns = tf.decode_csv(line, record_defaults=CSV_COLUMN_DEFAULTS)
        features = dict(zip(CSV_COLUMNS, columns))
        labels = features.pop(LABEL_COLUMN)
        return features, labels

    dataset = tf.data.TextLineDataset(filenames) # can pass one filename or filename list
    dataset = dataset.map(parse_csv, num_parallel_calls=10).prefetch(500000)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(num_epochs).batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels
