#!/usr/bin/env python
#coding=utf-8
'''
1. referce deep &wide, so we only need to write model_fn and consturt attention fm Estimator

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
    dropout = map(float, params["dropout"].split(','))

    #------bulid weights------
    Global_Bias = tf.get_variable(name='bias', shape=[1], initializer=tf.constant_initializer(0.0))
    Feat_Bias = tf.get_variable(name='linear', shape=[feature_size], initializer=tf.glorot_normal_initializer())
    Feat_Emb = tf.get_variable(name='emb', shape=[feature_size,embedding_size], initializer=tf.glorot_normal_initializer())

    #------build feaure-------
    feat_ids  = features['feat_ids']
    feat_ids = tf.reshape(feat_ids,shape=[-1,field_size])
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals,shape=[-1,field_size])

    #------build f(x)------
    with tf.variable_scope("Linear-part"):
        feat_wgts = tf.nn.embedding_lookup(Feat_Bias, feat_ids) # None * F * 1
        y_linear = tf.reduce_sum(tf.multiply(feat_wgts, feat_vals),1)

    with tf.variable_scope("Pairwise-Interaction-Layer"):
        embeddings = tf.nn.embedding_lookup(Feat_Emb, feat_ids) # None * F * K
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])
        embeddings = tf.multiply(embeddings, feat_vals) #vij*xi

        num_interactions = field_size*(field_size-1)/2
        element_wise_product_list = []
        for i in range(0, field_size):
            for j in range(i+1, field_size):
                element_wise_product_list.append(tf.multiply(embeddings[:,i,:], embeddings[:,j,:]))
        element_wise_product = tf.stack(element_wise_product_list) 								# (F*(F-1)) * None * K
        element_wise_product = tf.transpose(element_wise_product, perm=[1,0,2]) 				# None * (F*(F-1)) * K

    with tf.variable_scope("Attention-part"):
        deep_inputs = tf.reshape(element_wise_product, shape=[-1, embedding_size]) 				# (None * (F*(F-1))) * K

        aij = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity, \
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='attention_out')# (None * (F*(F-1))) * 1

        #aij_reshape = tf.reshape(aij, shape=[-1, num_interactions, 1])							# None * (F*(F-1)) * 1
        aij_softmax = tf.nn.softmax(tf.reshape(aij, shape=[-1, num_interactions, 1]), dim=1, name='attention_soft')
        if mode == tf.estimator.ModeKeys.TRAIN:
            aij_softmax = tf.nn.dropout(aij_softmax, keep_prob=dropout[0])

    with tf.variable_scope("Attention-based-Pooling"):
        y_emb = tf.reduce_sum(tf.multiply(aij_softmax, element_wise_product), 1) 				# None * K
        if mode == tf.estimator.ModeKeys.TRAIN:
            y_emb = tf.nn.dropout(y_emb, keep_prob=dropout[1])

        y_d = tf.contrib.layers.fully_connected(inputs=y_emb, num_outputs=1, activation_fn=tf.identity, \
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='deep_out')		# None * 1
        y_deep = tf.reshape(y_d,shape=[-1])

    with tf.variable_scope("AFM-out"):
        y_bias = Global_Bias * tf.ones_like(y_deep, dtype=tf.float32)   # None * 1
        y = y_bias + y_linear + y_deep
        pred = tf.sigmoid(y)

    predictions={"prob": pred}
    export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs)

    #------build loss------
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels)) + \
        l2_reg * tf.nn.l2_loss(Feat_Bias) + l2_reg * tf.nn.l2_loss(Feat_Emb)

    eval_metric_ops = {
        "auc": tf.metrics.auc(labels, pred)
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                eval_metric_ops=eval_metric_ops)
        
    #------build optimizer------
    #TODO:move optimizer to common function
    if FLAGS.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif FLAGS.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif FLAGS.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op)

