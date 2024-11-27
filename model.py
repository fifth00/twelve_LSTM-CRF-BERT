#!/usr/bin/env python
# encoding: utf-8
"""
@author: jfjiang6
@file: model.py

"""
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


# todo: LSTM+CRF模型构建：继承tf.keras.Model类来自定义LstmCrf模型类
# 在继承类中，需要重写 __init__()（构造函数，初始化）和 call()（模型调用）两个方法
class LstmCrfModel(tf.keras.Model):
    def __init__(self, hidden_size, vocab_size, label_size, embed_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.transition_params = None

        self.embedding = Embedding(vocab_size, embed_dim)  # 词嵌入
        self.LSTM = LSTM(hidden_size, return_sequences=True)  # LSTM
        self.dense = Dense(label_size)  # 全连接
        self.dropout = Dropout(0.5)  # 正则化，忽略一半隐层节点
        self.transition_params = tf.Variable(
            tf.random.uniform(shape=(self.label_size, self.label_size)), trainable=False)

    def call(self, text, labels=None, training=None):  # 设置labels=None，方便预测时使用
        text_lens = tf.math.reduce_sum(tf.cast(tf.math.not_equal(text, 0), dtype=tf.int32), axis=-1)
        inputs = self.embedding(text)
        inputs = self.dropout(inputs, training)
        inputs = self.LSTM(inputs)
        logits = self.dense(inputs)

        if labels is not None:
            label_sequences = tf.convert_to_tensor(labels)
            log_likelihood, self.transition_params =\
                tfa.text.crf_log_likelihood(logits, label_sequences, text_lens, self.transition_params)
            return logits, text_lens, log_likelihood
        else:
            return logits, text_lens
