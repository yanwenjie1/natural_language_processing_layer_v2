# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2022/9/7
@Time    : 8:45
@File    : process.py
@Function: XX
@Other: XX
"""
import os
import json
import sys
# 获取当前工作目录的路径
# 将当前工作目录添加到sys环境变量中
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
for it in sys.path:
    print('it  ', it)
from tqdm import tqdm
import pickle
from transformers import BertTokenizerFast
import config
import random
import re
from bs4 import BeautifulSoup
import torch
from utils.train_models import LayerFeature


def write_label(filename):
    results_labels = []
    with open(filename, encoding='utf-8') as file:
        contents = json.load(file)
    contents = sorted(contents, key=lambda x: x['id'])
    for content in tqdm(contents):
        assert len(content['annotations']) == 1, '存在多人标注结果'
        assert len(content['drafts']) <= 1, '存在多人缓存结果' + str(len(content['drafts']))
        annotations = content['annotations'][0]
        labels = annotations['result']
        if len(content['drafts']) >= 1:
            labels2_list = [i['result'] for i in content['drafts']]
            for iii in labels2_list:
                labels.extend(iii)
            # labels2_list = sorted(labels2_list, key=len)
            # labels.extend(labels2_list[-1])
        labels = [{
            # 'id': i['id'],
            'start': i['value']['start'],
            'end': i['value']['end'],
            'labels': i['value']['hypertextlabels']
            }for i in labels]
        for item in labels:
            for one_label in item['labels']:
                if one_label not in results_labels:
                    results_labels.append(one_label)
    results_labels = ['S-' + i for i in results_labels]
    results_labels.sort()
    results_labels.insert(0, 'O')
    with open('labels.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(results_labels, ensure_ascii=False))
    return results_labels


def load_data_table(filename):
    results = []
    labels_to_ids = {j: i for i, j in enumerate(all_entity_labels)}
    with open(filename, encoding='utf-8') as file:
        contents = json.load(file)
    contents = sorted(contents, key=lambda x: x['id'])
    wrong_list = []
    for content in tqdm(contents):
        try:
            # if content['id'] == 357654:
            #     pass
            html = content['data']['html']
            soup = BeautifulSoup(html, 'html.parser')
            lines = soup.find_all('p')
            this_lines = [i.text.strip() for i in lines]
            this_labels = [0] * len(this_lines)
            labels = content['annotations'][0]['result']

            labels = [{
                # 'id': i['id'],
                'start': i['value']['start'],
                'end': i['value']['end'],
                'text': i['value']['text'],
                'labels': i['value']['hypertextlabels']
            } for i in labels]

            for label in labels:
                assert len(label['labels']) == 1, '多标签标注，标签id：' + str(labels['id'])
                # '/p[705]/text()[1]'
                re_start = re.findall(r'p\[(\d+)\]', label['start'])
                re_end = re.findall(r'p\[(\d+)\]', label['end'])
                assert len(re_end) == len(re_start) == 1, '匹配标注内容起止失败，标签id：' + str(labels['id'])
                int_start = int(re_start[0])
                int_end = int(re_end[0])
                assert int_start == int_end, '匹配标注内容起止不同，标签id：' + str(labels['id'])
                this_labels[int_start] = labels_to_ids['S-' + label['labels'][0]]

            # 按照 max_seq_len 拼接语料(按照128)
            start = 0
            while start < len(this_lines):
                results.append((this_lines[start: start + args.max_seq_len], this_labels[start: start + args.max_seq_len]))
                start += args.max_seq_len - 20
        except Exception as e:
            print(str(e))
            wrong_list.append(content['id'])

    print(wrong_list)
    return results


def convert_examples_to_features(examples, tokenizer: BertTokenizerFast):
    features = []
    for (one_texts, one_labels) in tqdm(examples):  # texts: list of str; entities: list of tuple (from_ids, to_ids, label)
        # 如果这里不够 args.max_seq_len 那就要调整mask
        assert len(one_labels) == len(one_texts) <= args.max_seq_len, '文本和标签长度未对齐'
        while len(one_texts) != args.max_seq_len:
            one_texts.append('空白数据填充')
            one_labels.append(0)
        masks = torch.zeros((args.max_seq_len,), dtype=torch.uint8)
        labels = torch.as_tensor(one_labels, dtype=torch.uint8)

        word_ids = tokenizer.batch_encode_plus(one_texts,
                                               max_length=args.max_word_len,
                                               padding="max_length",
                                               truncation=True,
                                               return_token_type_ids=True,
                                               return_attention_mask=True,
                                               return_tensors='pt')
        token_ids = word_ids['input_ids']
        attention_masks = word_ids['attention_mask'].byte()
        token_type_ids = word_ids['token_type_ids'].byte()

        for index_text, text in enumerate(one_texts):
            if text != '空白数据填充':
                masks[index_text] = 1

        feature = LayerFeature(
            token_ids=token_ids,
            attention_masks=attention_masks,
            token_type_ids=token_type_ids,
            masks=masks,
            labels=labels,
        )
        features.append(feature)

    return features


if __name__ == '__main__':
    args = config.Args().get_parser()
    #  调整配置
    args.data_dir = os.getcwd()
    args.max_seq_len = 128  # 滑窗视野为128
    args.max_word_len = 40  # 每行最多取40个token

    my_tokenizer = BertTokenizerFast.from_pretrained('../../' + args.bert_dir)
    dataPath = 'project-184-at-2024-01-26-03-21-1f0728f2.json'
    all_entity_labels = write_label(dataPath)

    all_data = load_data_table(dataPath)

    print('总样本量 ', len(all_data))

    random.shuffle(all_data)  # 打乱数据集 386条*1400行/128拼接=4400条语料
    train_data = all_data[int(len(all_data) / 7):]
    dev_data = all_data[:int(len(all_data) / 7) + 1]

    train_data = convert_examples_to_features(train_data, my_tokenizer)
    dev_data = convert_examples_to_features(dev_data, my_tokenizer)
    with open(os.path.join(args.data_dir, '{}.pkl'.format('train_data')), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(args.data_dir, '{}.pkl'.format('dev_data')), 'wb') as f:
        pickle.dump(dev_data, f)
