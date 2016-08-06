#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
import logging
import random
import numpy as np
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string("train_data", None, "training file.")
flags.DEFINE_string("test_data", None, "testing file.")
flags.DEFINE_string("save_path", None, "path to save the model.")
flags.DEFINE_integer("max_epoch", 10, "max epochs.")
flags.DEFINE_integer("emb_dim", 50, "embedding dimension.")
flags.DEFINE_float("lr", 0.025, "initial learning rate.")


FLAGS = flags.FLAGS


class Options(object):
    """options used by CDK model."""
    
    def __init__(self):
        #model options.

        #embedding dimension.
        self.emb_dim = FLAGS.emb_dim

        #train file path.
        self.train_data = FLAGS.train_data

        #test file path.
        self.test_data = FLAGS.test_data

        #save path.
        self.save_path = FLAGS.save_path

        #max epoch.
        self.max_epoch = FLAGS.max_epoch

        #initial learning rate.
        self.lr = FLAGS.lr


class Embedded_IC(object):
    """Embedded IC model."""

    def __init__(self, options, session):
        self._options = options
        self._session = session
        self._u2idx = {}
        self._buildIndex()
        self._train_cascades = self._readFromFile(options.train_data)
        self._test_cascades = self._readFromFile(options.test_data)
        self._options.train_size = len(self._train_cascades)
        self._options.test_size = len(self._test_cascades)
        logging.info("training set size:%d    testing set size:%d" % (self._options.train_size, self._options.test_size))
        self._options.samples_to_train = self._options.max_epoch * self._options.train_size
        self.buildGraph()


    def _buildIndex(self):
        #compute an index of the users that appear at least once in the training and testing cascades.
        opts = self._options

        train_user_set = set()
        test_user_set = set()

        for line in open(opts.train_data):
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split()
            for chunk in chunks:
                user, timestamp = chunk.split(',')
                train_user_set.add(user)

        for line in open(opts.test_data):
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split()
            for chunk in chunks:
                user, timestamp = chunk.split(',')
                test_user_set.add(user)

        user_set = train_user_set | test_user_set

        pos = 0
        for user in user_set:
            self._u2idx[user] = pos
            pos += 1
        opts.user_size = len(user_set)
        logging.info("user_size : %d" % (opts.user_size))


    def _readFromFile(self, filename):
        """read all cascade from training or testing files. """
        t_cascades = []
        for line in open(filename):
            if len(line.strip()) == 0:
                continue
            userlist = []
            chunks = line.strip().split()
            for chunk in chunks:
                user, timestamp = chunk.split(',')
                if self._u2idx.has_key(user):
                    userlist.append(self._u2idx[user])

            if len(userlist) > 1:
                t_cascades.append(userlist)

        return t_cascades


    def buildGraph(self):
        opts = self._options
