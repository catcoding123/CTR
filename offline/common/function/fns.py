#!/usr/bin/env python
#coding=utf-8
'''
utility function
'''

# TODO: use tensorflow keys
def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True, updates_collections=None, is_training=True,  reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True, updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z
