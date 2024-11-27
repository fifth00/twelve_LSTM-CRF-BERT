#!/usr/bin/env python
# encoding: utf-8
"""
@author: jfjiang6
@file: ner_lstmcrf_train.py

最新版的tensorflow2.1默认安装cpu和gpu两个版本，gpu不能运行时退回到cpu版本

TensorFlow2.0和2.1中暂无CRF的实现，因此要通过 install tensorflow.addons 来加载CRF
tensorflow.addons 版本为0.8.2 与tensorflow2.0版本不兼容，需要TensorFlow2.1版本

"""
import os
import time
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import Adam
from data_process import read_corpus  # 导入语料读取函数
from model import LstmCrfModel  # 导入 LSTM+CRF 模型类
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 设置只使用CPU训练


# #############################################################################################
# 步骤1 todo: 语料数据处理（标记类别及字符词典处理）
# 步骤1.1 todo: 加载命名实体识别的BIO标记类别，并将BIO标签与索引对应
# "BIO"标记：B-PER和I-PER表示人名，B-LOC和I-LOC表示地名，B-ORG和I-ORG表示机构名
label2idx = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6}
idx2label = {idx: label for label, idx in label2idx.items()}  # 将索引与BIO标签对应

# 步骤1.2 todo: 读取字符集文件，并将字符与索引编号对应
# 步骤1.2.1 todo: 读取字符词典文件
special_words = ['<PAD>', '<UNK>']  # 特殊词表示
with open('./data/char_vocabs.txt', 'r', encoding='utf-8') as fo:
    char_vocabs = [line.strip() for line in fo]
char_vocabs = special_words + char_vocabs
# 步骤1.2.2 todo: 将字符与索引编号对应
idx2vocab = {idx: char for idx, char in enumerate(char_vocabs)}
vocab2idx = {char: idx for idx, char in idx2vocab.items()}


# #############################################################################################
# 步骤2 todo: 获取训练集的文本和标注，并进行序列填充
# 步骤2.1 todo: 加载训练集的文本和标注
# 使用的数据是已经预处理过的，所以直接加载数据
train_text, train_labels = read_corpus('./data/train.txt', vocab2idx, label2idx)

# 步骤2.2 todo: 查看数据集的信息
print('训练集中第1个文本：\n', train_text[0])
print('训练集中第1个文本对应的编号：\n', [idx2vocab[idx] for idx in train_text[0]])
print('训练集中第1个文本的标注：\n', train_labels[0])
print('训练集中第1个文本标签对应的编号：\n', [idx2label[idx] for idx in train_labels[0]])

# 步骤2.3 todo: 将训练集的文本和标注序列进行相同长度填充（LSTM要求长度一致）
max_len = 100
train_text = sequence.pad_sequences(train_text, maxlen=max_len, padding='post')
train_labels = sequence.pad_sequences(train_labels, maxlen=max_len, padding='post')
print('训练集文本的维度:', train_text.shape)
print('训练集标注的维度:', train_labels.shape)

# 步骤2.4 todo: 将训练集（文本和标注）按照batch_size分批
batch_size = 64  # 每批训练数据的大小
train_dataset = tf.data.Dataset.from_tensor_slices((train_text, train_labels))
train_dataset = train_dataset.shuffle(len(train_text)).batch(batch_size, drop_remainder=True)

# #############################################################################################
# 步骤3 todo: 模型训练及保存
# 步骤3.1 todo: 模型参数初始化
embed_dim = 300  # 词嵌入的维度
hidden_size = 256  # 隐藏层的数目
vocab_size = len(vocab2idx)  # 字符集的大小
label_size = len(label2idx)  # 标注的标签数目
num_epochs = 100  # 轮巡整个训练集的遍数
learning_rate = 0.001  # 优化算法的学习率

# 步骤3.2 todo: 实例化模型并实例化一个优化器（这里使用常用的 Adam 优化器）
model = LstmCrfModel(hidden_size, vocab_size, label_size, embed_dim)
optimizer = Adam(learning_rate=learning_rate)

# 步骤3.3 todo: 模型保存事先声明
ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)  # 首先声明一个 Checkpoint
ckpt_manager = tf.train.CheckpointManager(ckpt, 'checkpoints/', checkpoint_name='model.ckpt', max_to_keep=10)

# 步骤3.4 todo: 进行num_epochs次数据集轮巡的模型训练及模型保存
startTime = time.time()
best_acc = 0
step = 0  # 用于记录迭代次数，便于过程输出
for epoch in range(num_epochs):
    for _, (text_batch, labels_batch) in enumerate(train_dataset):
        step = step + 1
        with tf.GradientTape() as tape:
            logits, text_lens, log_likelihood = model(text_batch, labels_batch, training=True)
            loss = - tf.reduce_mean(log_likelihood)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if step % 20 == 0:
            paths = []
            accuracy = 0
            for logit, text_len, labels in zip(logits, text_lens, labels_batch):
                viterbi_path, _ = tfa.text.viterbi_decode(logit[:text_len], model.transition_params)
                paths.append(viterbi_path)
                correct_prediction = tf.equal(
                    tf.convert_to_tensor(sequence.pad_sequences([viterbi_path], padding='post')),
                    tf.convert_to_tensor(sequence.pad_sequences([labels[:text_len]], padding='post')))
                accuracy = accuracy + tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy = accuracy / len(paths)
            print('epoch %d, step %d, loss %.4f , accuracy %.4f' % (epoch, step, loss, accuracy))
            if accuracy > best_acc:
                best_acc = accuracy
                ckpt_manager.save()
print('模型训练(%d epoch, %d step)耗时： %.3f s' % (num_epochs, step, time.time() - startTime))

