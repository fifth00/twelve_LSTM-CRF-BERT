#!/usr/bin/env python
# encoding: utf-8
"""
@author: jfjiang6
@file: data_process.py

"""


# todo: 定义函数来读取训练语料和测试语料，获取文本和标注
# def read_corpus(corpus_path, vocab2idx, label2idx):
#     datas, labels = [], []
#     with open(corpus_path, encoding='utf-8') as fr:
#         lines = fr.readlines()
#     sent_, tag_ = [], []
#     for line in lines:
#         if line != '\n':
#             [char, label] = line.strip().split()
#             sent_.append(char)
#             tag_.append(label)
#         else:
#             sent_ids = [vocab2idx[char] if char in vocab2idx else vocab2idx['<UNK>'] for char in sent_]
#             tag_ids = [label2idx[label] if label in label2idx else 0 for label in tag_]
#             datas.append(sent_ids)
#             labels.append(tag_ids)
#             sent_, tag_ = [], []
#     return datas, labels
def read_corpus(corpus_path, vocab2idx, label2idx):
    datas, labels = [], []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()

    sent_, tag_ = [], []

    for line in lines:
        line = line.strip()  # 去掉两端的空白字符
        if not line:  # 如果是空行，跳过
            if sent_:
                # 将当前句子的字符和标签列表转化为索引，并添加到最终数据列表
                sent_ids = [vocab2idx.get(char, vocab2idx['<UNK>']) for char in sent_]
                tag_ids = [label2idx.get(label, label2idx['O']) for label in tag_]  # 假设 'O' 是默认标签
                datas.append(sent_ids)
                labels.append(tag_ids)
                sent_, tag_ = [], []  # 清空当前句子的临时列表
        else:
            try:
                char, label = line.split()  # 这里尝试拆分行
                sent_.append(char)
                tag_.append(label)
            except ValueError:
                print(f"跳过格式错误的行: {line}")
                continue  # 如果拆分失败，跳过这一行

    # 处理最后一个句子（如果文件末尾没有空行）
    if sent_:
        sent_ids = [vocab2idx.get(char, vocab2idx['<UNK>']) for char in sent_]
        tag_ids = [label2idx.get(label, label2idx['O']) for label in tag_]
        datas.append(sent_ids)
        labels.append(tag_ids)

    return datas, labels


# todo: 定义函数将列表的列表转化为列表: [[t1, t2], [t3, t4]...] --> [t1, t2, t3, t4...]
def flatten_lists(lists):
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list


# todo: 定义函数进行命名实体解析（解析成具体的人名、地名、机构名）和提取
# def get_valid_nertag(input_data, result_tags):
#     NER_words = []
#     start, end = 0, 1
#     tag_label = "O"
#     for i, tag in enumerate(result_tags):
#         if tag.startswith("B"):
#             if tag_label != "O":
#                 NER_words.append((input_data[start: end], tag_label))
#             tag_label = tag.split("-")[1]
#             start, end = i, i + 1
#         elif tag.startswith("I"):
#             temp_label = tag.split("-")[1]
#             if temp_label == tag_label:
#                 end += 1
#         elif tag == "O":
#             if tag_label != "O":
#                 NER_words.append((input_data[start: end], tag_label))
#                 tag_label = "O"
#             start, end = i, i + 1
#     if tag_label != "O":
#         NER_words.append((input_data[start: end], tag_label))
#     return NER_words
def get_valid_nertag(input_data, result_tags):
    NER_words = []
    start, end = 0, 1
    tag_label = "O"
    for i, tag in enumerate(result_tags):
        tag = str(tag)  # 确保tag是字符串类型
        if tag.startswith("B"):
            if tag_label != "O":
                NER_words.append((input_data[start: end], tag_label))
            tag_label = tag.split("-")[1]
            start, end = i, i + 1
        elif tag.startswith("I"):
            temp_label = tag.split("-")[1]
            if temp_label == tag_label:
                end += 1
        elif tag == "O":
            if tag_label != "O":
                NER_words.append((input_data[start: end], tag_label))
                tag_label = "O"
            start, end = i, i + 1
    if tag_label != "O":
        NER_words.append((input_data[start: end], tag_label))
    return NER_words


#
# # todo: 定义函数进行命名实体解析（解析成具体的人名、地名、机构名）和提取
# def get_valid_nertag(input_data, result_tags):
#     NER_words = []
#     start, end = 0, 1  # 实体开始结束位置标识
#     tag_label = "O"  # 实体类型标识
#     for i, tag in enumerate(result_tags):
#         if tag.startswith("B"):
#             if tag_label != "O":  # 当前实体tag之前有其他实体
#                 NER_words.append((input_data[start: end], tag_label))  # 获取实体
#             tag_label = tag.split("-")[1]  # 获取当前实体类型
#             start, end = i, i + 1  # 开始和结束位置变更
#         elif tag.startswith("I"):
#             temp_label = tag.split("-")[1]
#             if temp_label == tag_label:  # 当前实体tag是之前实体的一部分
#                 end += 1  # 结束位置end扩展
#         elif tag == "O":
#             if tag_label != "O":  # 当前位置非实体 但是之前有实体
#                 NER_words.append((input_data[start: end], tag_label))  # 获取实体
#                 tag_label = "O"  # 实体类型置"O"
#             start, end = i, i + 1  # 开始和结束位置变更
#     if tag_label != "O":  # 最后结尾还有实体
#         NER_words.append((input_data[start: end], tag_label))  # 获取结尾的实体
#     return NER_words

# if __name__ == '__main__':
#     # 假设已经定义了 read_corpus, flatten_lists, get_valid_nertag 函数
#
#     # 示例 vocab2idx 和 label2idx 字典
#     vocab2idx = {'<UNK>': 0, '<PAD>': 1, 'bl': 2, '文': 3, '强': 4, '推': 5, 'P': 6, '大': 7, '的': 8, '小': 9,
#                  '说': 10, ',': 11, 'priest': 12, '哥': 13, '全': 14, '名': 15, '六': 16, '爻': 17, '镇': 18, '魂': 19,
#                  '杀': 20, '破': 21, '狼': 22, '部': 23, '精': 24}
#     label2idx = {'O': 0, 'B-book': 1, 'I-book': 2}  # 这里只是一个示例，你可以根据实际数据补充更多标签
#
#     # 使用 read_corpus 函数读取并处理数据
#     corpus_path = 'E:/NLM/1_基于LSTM+CRF的命名实体识别/1_基于LSTM+CRF的命名实体识别/01、实验代码/data/test.txt'  # 根据实际文件路径修改
#     datas, labels = read_corpus(corpus_path, vocab2idx, label2idx)
#
#     # 使用 flatten_lists 将 datas 和 labels 展平
#     flat_datas = flatten_lists(datas)
#     flat_labels = flatten_lists(labels)
#
#     # 使用 get_valid_nertag 解析命名实体
#     # 注意，这里 input_data 是展平后的字符数据，result_tags 是展平后的标签数据
#     ner_results = get_valid_nertag(flat_datas, flat_labels)
#
#     # 打印解析出的命名实体
#     print("命名实体解析结果：")
#     for entity, tag in ner_results:
#         print(f"实体: {''.join(entity)}, 标签: {tag}")
#
#     # 生成最终的 text_data
#     # 你可以根据需求将每个命名实体信息组织成一个更符合需求的格式
#     text_data = []
#     for entity, tag in ner_results:
#         # 假设我们希望以字典的形式存储每个实体及其标签
#         text_data.append({"entity": ''.join(entity), "label": tag})
#
#     # 打印最终的 text_data
#     print("最终生成的 text_data：")
#     for item in text_data:
#         print(item)
