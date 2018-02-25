#!/usr/bin/env python
#coding=utf-8
'''
1. referce deep &wide, so we only need to write model_fn and consturt deep cross network Estimator

'''

import shutil
import sys
import os
import json
from datetime import date, timedelta
from time import time
import random
import pandas as pd
import numpy as np
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/common/')
sys.path.append(os.getcwd())
from function import *

def model_fn(features, labels, mode, params):

    #------hyperparameters----
    field_size = params["field_size"]
    feature_size = params["feature_size"]
    embedding_size = params["embedding_size"]
    l2_reg = params["l2_reg"]
    learning_rate = params["learning_rate"]
    layers = map(int, params["deep_layers"].split(','))
    dropout = map(float, params["dropout"].split(','))
    cross_layer_num = params["cross_layer_num"]
    total_size = field_size * embedding_size

    #------bulid weights------
    GLOBAL_B = tf.get_variable(name='global_bias', shape=[1], initializer=tf.constant_initializer(0.0))
    GLOBAL_W = tf.get_variable(name='global_embeddings', shape=[feature_size, embedding_size], initializer=tf.glorot_normal_initializer())
    CROSS_W = [tf.get_variable(name='cross_weight_%d' %i, shape=[total_size, 1], initializer=tf.glorot_normal_initializer()) for i in range(cross_layer_num)] 
    CROSS_B = [tf.get_variable(name='cross_bias_%d' %i, shape=[total_size, 1], initializer=tf.glorot_normal_initializer()) for i in range(cross_layer_num)] 

    #------build feaure-------
    feat_ids  = features['feat_ids']
    feat_ids = tf.reshape(feat_ids,shape=[-1,field_size])
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals,shape=[-1,field_size])

    #------build f(x)------
    with tf.variable_scope("Cross-part"):
        embeddings = tf.nn.embedding_lookup(GLOBAL_W, feat_ids) # None * F * K
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])
        embeddings = tf.multiply(embeddings, feat_vals) #vij*xi
        x0 = tf.reshape(embeddings, shape=[-1, total_size, 1])
        x_l = x0
        for l in range(cross_layer_num):
            x_l = tf.matmul(self._x_0, x_l, transpose_b=True)
            x_l = tf.tensordot(x_l, CROSS_W[l], 1) + CROSS_B[l] + x_l
        cross_output = tf.reshape(x_l, shape=[-1, total_size])

    with tf.variable_scope("Deep-part"):
        if FLAGS.batch_norm:
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_phase = True
            else:
                train_phase = False
        else:
            normalizer_fn = None
            normalizer_params = None

        deep_inputs = tf.reshape(embeddings,shape=[-1,field_size*embedding_size]) # None * (F*K)
        for i in range(len(layers)):
            deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers[i], \
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='mlp%d' % i)
            if FLAGS.batch_norm:
                deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase, scope_bn='bn_%d' %i)  
            if mode == tf.estimator.ModeKeys.TRAIN:
                deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[i])                              
        deep_output = deep_inputs

    with tf.variable_scope("combine-part"):
        x_combine = tf.concat([cross_output, deep_output],1)
        y_combine = tf.contrib.layers.fully_connected(inputs=x_combine, num_outputs=1, activation_fn=tf.identity, \
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='combine_out')
        y_combine = tf.reshape([y_combine, shape=[-1]])
        y_bias = GLOBAL_B * tf.ones_like(x, dtype=tf.float32)     # None * 1
        y = y_combine + y_bias
        pred = tf.sigmoid(y)

    predictions={"prob": pred}
    export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs)

    #------build loss------
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels)) + \
        l2_reg * tf.nn.l2_loss(FM_W) + \
        l2_reg * tf.nn.l2_loss(FM_V) #+ \ l2_reg * tf.nn.l2_loss(sig_wgts)

    # Provide an estimator spec for `ModeKeys.EVAL`
    eval_metric_ops = {
        "auc": tf.metrics.auc(labels, pred)
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                eval_metric_ops=eval_metric_ops)



            
