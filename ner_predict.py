#!/usr/bin/env python
# encoding: utf-8
"""
@author: jfjiang6
@file: ner_predict.py

"""

import os
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import Adam
from collections import Counter
from data_process import read_corpus, flatten_lists, get_valid_nertag
from model import LstmCrfModel
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 设置只使用CPU训练
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# #############################################################################################
# todo: 构建类别标注的评价体系并存储在evaluation中，全局变量
evaluation = pd.DataFrame({'tag': [],
                           '精确率': [],
                           '召回率': [],
                           ' F1 值': [],
                           'support数目': [],
                           'predict数目': []})


# #############################################################################################
# 步骤1 todo: 基础数据准备（与训练文件中一致）
# "BIO"标记：B-PER和I-PER表示人名，B-LOC和I-LOC表示地名，B-ORG和I-ORG表示机构名
label2idx = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6}
idx2label = {idx: label for label, idx in label2idx.items()}  # 将索引与BIO标签对应
special_words = ['<PAD>', '<UNK>']  # 特殊词表示
with open('./data/char_vocabs.txt', 'r', encoding='utf-8') as fo:
    char_vocabs = [line.strip() for line in fo]
char_vocabs = special_words + char_vocabs
idx2vocab = {idx: char for idx, char in enumerate(char_vocabs)}
vocab2idx = {char: idx for idx, char in idx2vocab.items()}

# #############################################################################################
# 步骤2 todo: 模型恢复
# 步骤2.1 todo: 参数初始化，与训练时一致
label_size = len(label2idx)  # 标注的标签数目
vocab_size = len(vocab2idx)  # 字符集的大小
embed_dim = 300  # 词嵌入的维度
hidden_size = 256  # 隐藏层的数目
learning_rate = 0.001  # 优化算法的学习率

# 步骤2.2 todo: 模型恢复
model = LstmCrfModel(hidden_size, vocab_size, label_size, embed_dim)
optimizer = Adam(learning_rate=learning_rate)
ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
ckpt.restore(tf.train.latest_checkpoint('checkpoints/', latest_filename='model.ckpt-9'))

# #############################################################################################
# 步骤3 todo: 测试集上的命名实体识别预测及评价
# 步骤3.1 todo: 加载测试集的文本和标注，对序列进行相同长度填充
test_text, test_labels = read_corpus('./data/test.txt', vocab2idx, label2idx)
max_len = 100
test_text = sequence.pad_sequences(test_text, maxlen=max_len, padding='post')
print('测试集文本的维度:', test_text.shape)

# 步骤3.2 todo: 模型预测
test_logits, test_text_lens = model.predict(test_text)

# 步骤3.3 todo: 第一种评价方法：模型预测精确度计算
test_labels_paths = []  # 用于存储标注索引
test_accuracy = 0
for logit, text_len, label in zip(test_logits, test_text_lens, test_labels):
    viterbi_path, _ = tfa.text.viterbi_decode(logit[:text_len], model.transition_params)  # viterbi算法解码
    test_labels_paths.append(viterbi_path)
    correct_prediction = tf.equal(tf.convert_to_tensor(sequence.pad_sequences([viterbi_path], padding='post')),
                                  tf.convert_to_tensor(sequence.pad_sequences([label[:text_len]], padding='post')))
    test_accuracy = test_accuracy + tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_accuracy = test_accuracy / len(test_labels_paths)
print('模型在测试集上的准确率为：%.4f' % test_accuracy)

# 步骤3.4 todo: 第二种评价方法：计算每个标注类别及总体的精确率、召回率和F1值并保存
# 步骤3.4.1 todo: 将标注数据列表进行格式转化(列表的列表转化为列表）
test_labels_lists = flatten_lists(test_labels)
test_labels_pre_lists = flatten_lists(test_labels_paths)

# 步骤3.4.2 todo: 统计标注数据列表（测试集和预测所得集）中的标注类别（类别及对应的个数）
test_labels_counter = Counter(test_labels_lists)
test_labels_pre_counter = Counter(test_labels_pre_lists)

# 步骤3.4.3 todo: 计算每个标注类别预测正确的个数，用于后面精确率以及召回率的计算
correct_tags_number = {}
for test_tag, predict_tag in zip(test_labels_lists, test_labels_pre_lists):
    if test_tag == predict_tag:
        if test_tag not in correct_tags_number:
            correct_tags_number[test_tag] = 1
        else:
            correct_tags_number[test_tag] += 1

# 步骤3.4.4 todo: 计算每个标注类别的精确率、召回率和F1值
precision_scores = {}  # 精确率
recall_scores = {}  # 召回率
f1_scores = {}  # F1值
test_labels_set = set(test_labels_lists)  # 将列表转化为集合（去掉了重复的标注类别），计算效率高
for tag in test_labels_set:
    precision_scores[tag] = correct_tags_number.get(tag, 0) / test_labels_pre_counter[tag]
    recall_scores[tag] = correct_tags_number.get(tag, 0) / test_labels_counter[tag]
    f1_scores[tag] = 2 * precision_scores[tag] * recall_scores[tag] \
                     / (precision_scores[tag] + recall_scores[tag] + 1e-10)  # 分母加上很小的数，防止为0

# 步骤3.4.5 todo: 计算所有标注类别（不包含O，加权平均）的精确率、召回率和F1值
weighted_ave = {'precision': 0., 'recall': 0., 'f1_score': 0.}
total = len(test_labels_lists)
for tag in test_labels_set:
    if tag is not 0:  # 不包含标记为0的情况
        size = test_labels_counter[tag]
        weighted_ave['precision'] += precision_scores[tag] * size
        weighted_ave['recall'] += recall_scores[tag] * size
        weighted_ave['f1_score'] += f1_scores[tag] * size

for key in weighted_ave.keys():
    weighted_ave[key] /= (total-test_labels_counter[0])

# 步骤3.4.6 todo: 将评价结果存入evaluation中
# 步骤3.4.6.1 todo: 将每个标注类别的精确率、召回率和F1值及对应的标注个数存入evaluation中
for tag in test_labels_set:
    r = evaluation.shape[0]
    evaluation.loc[r] = [tag,
                         format(precision_scores[tag], '.5f'),
                         format(recall_scores[tag], '.5f'),
                         format(f1_scores[tag], '.5f'),
                         test_labels_counter[tag],
                         test_labels_pre_counter[tag]]

# 步骤3.4.6.2 todo: 将总体标注类别（不包含O）的精确率、召回率和F1值存入evaluation中
r = evaluation.shape[0]
evaluation.loc[r] = ['avg/total',
                     format(weighted_ave['precision'], '.5f'),
                     format(weighted_ave['recall'], '.5f'),
                     format(weighted_ave['f1_score'], '.5f'),
                     len(test_labels_lists)-test_labels_counter[0],
                     len(test_labels_pre_lists)-test_labels_pre_counter[0]]

# 步骤3.4.7 todo: 将评价指标evaluation保存，encoding防止中文乱码
evaluation.to_csv('./evaluation2.csv', sep=',', header=True, index=True, encoding='utf_8_sig')


# #############################################################################################
# 步骤4 todo: 输入文本命名实体识别预测
# 步骤4.1 todo: 输入文本，并进行相应的序列处理
# text = input("请输入带有人名、地名或机构的文本:")
text = "中华人民共和国国务院总理周恩来在外交部长陈毅的陪同下，连续访问了埃塞俄比亚以及阿尔巴尼亚等非洲10国"
text_chars = list(text)
dataset = sequence.pad_sequences([[vocab2idx.get(char, 0) for char in text]], padding='post')
print('输入的文本转化后的序列为：\n', dataset)

# 步骤4.2 todo: 模型预测
logits, text_lens = model.predict(dataset)

# 步骤4.3 todo: 预测后的标注解码
paths = []  # 用于存储标注索引
for logit, text_len in zip(logits, text_lens):
    viterbi_path, _ = tfa.text.viterbi_decode(logit[:text_len], model.transition_params)
    paths.append(viterbi_path)
label_pred = [idx2label[idx] for idx in paths[0]]
print('输入的文本预测后的标注索引为:\n', paths[0])
print('输入的文本预测后的标注为:\n', label_pred)

# 步骤4.4 todo: 对输入文本预测后的标注进行命名实体解析（解析成具体的人名、地名、机构名）和提取并显示
NER_words = get_valid_nertag(text_chars, label_pred)
for (word, tag) in NER_words:
    print("".join(word), tag)

