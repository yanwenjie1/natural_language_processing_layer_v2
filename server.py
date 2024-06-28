# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/5/9
@Time    : 13:12
@File    : server_confindence.py
@Function: XX
@Other: XX
"""
import json
import os
import socket
import numpy as np
from tqdm import trange
from flask import Flask, request
from gevent import pywsgi
from transformers import BertTokenizerFast
from utils.functions import load_model_and_parallel, get_entity_bieos
from utils.train_models import LayerLabeling
from pathlib import Path
import onnxruntime
import traceback


def torch_env():
    """
    测试torch环境是否正确
    :return:
    """
    import torch.backends.cudnn

    print('torch版本:', torch.__version__)  # 查看torch版本
    print('cuda版本:', torch.version.cuda)  # 查看cuda版本
    print('cuda是否可用:', torch.cuda.is_available())  # 查看cuda是否可用
    print('可行的GPU数目:', torch.cuda.device_count())  # 查看可行的GPU数目 1 表示只有一个卡
    print('cudnn版本:', torch.backends.cudnn.version())  # 查看cudnn版本
    print('输出当前设备:', torch.cuda.current_device())  # 输出当前设备（我只有一个GPU为0）
    print('0卡名称:', torch.cuda.get_device_name(0))  # 获取0卡信息
    print('0卡地址:', torch.cuda.device(0))  # <torch.cuda.device object at 0x7fdfb60aa588>
    x = torch.rand(3, 2)
    print('onnxruntime环境:', onnxruntime.get_device())
    print(x)  # 输出一个3 x 2 的tenor(张量)


def get_ip_config():
    """
    ip获取
    :return:
    """
    myIp = [item[4][0] for item in socket.getaddrinfo(socket.gethostname(), None) if ':' not in item[4][0]][0]
    return myIp


def encode(texts: list[str]):
    """
    :param texts: list of str
    :return:
    """
    assert type(texts) == list
    results = []

    # 按照 max_seq_len 拼接语料
    start = 0
    while start < len(texts):
        texts_part = texts[start: start + args.max_seq_len]
        while len(texts_part) < args.max_seq_len:
            texts_part.append('空白补充-PAD')
        masks = np.ones((1, len(texts_part)), dtype=bool)
        word_ids = tokenizer.batch_encode_plus(texts_part,
                                               max_length=args.max_word_len,
                                               padding='max_length',
                                               truncation=True,
                                               return_token_type_ids=True,
                                               return_attention_mask=True,
                                               return_tensors='np')
        token_ids = np.expand_dims(word_ids['input_ids'], axis=0)
        attention_masks = np.expand_dims(word_ids['attention_mask'], axis=0)
        token_type_ids = np.expand_dims(word_ids['token_type_ids'], axis=0)

        for i, j in enumerate(texts_part):
            if j == "空白补充-PAD":
                masks[0, i] = False

        assert token_ids.shape == (1, args.max_seq_len, args.max_word_len), f'token_ids.size不对{str(token_ids.shape)}'
        assert attention_masks.shape == (1, args.max_seq_len, args.max_word_len), f'token_ids.size不对{str(attention_masks.shape)}'
        assert token_type_ids.shape == (1, args.max_seq_len, args.max_word_len), f'token_ids.size不对{str(token_type_ids.shape)}'
        assert masks.shape == (1, args.max_seq_len), f'token_ids.size不对{str(masks.shape)}'

        results.append((token_ids, attention_masks, token_type_ids, masks))
        if len(texts_part) < args.max_seq_len or start + args.max_seq_len >= len(texts):
            break
        start += args.max_seq_len - 20

    return results


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    softmax_x = exp_x / sum_exp_x
    return softmax_x


def decode(encodings, len_sentence):
    """

    :param token_ids:
    :param attention_masks:
    :param token_type_ids:
    :return:
    """
    global_index = 0  # 指示第几个滑窗
    global_indexs = np.zeros((len_sentence,), dtype=np.int16)
    global_values = np.zeros((len_sentence,), dtype=np.float16)
    step = 8
    # 默认取batch推理为8 对输入的encodings做batch为8的切片
    for i in range(0, len(encodings), step):
        encodings_part = encodings[i: i+step]
        token_ids = np.concatenate([j[0] for j in encodings_part], axis=0, dtype=np.int32)
        attention_masks = np.concatenate([j[1] for j in encodings_part], axis=0, dtype=np.int32)
        token_type_ids = np.concatenate([j[2] for j in encodings_part], axis=0, dtype=np.int32)
        masks = np.concatenate([j[3] for j in encodings_part], axis=0)

        logits = onnx_model.run(None, {   
                                    "token_ids": token_ids, 
                                    "attention_masks": attention_masks, 
                                    "token_type_ids":  token_type_ids})[0]


        logits = softmax(logits)
        values = np.max(logits, axis=-1)
        indexs = np.argmax(logits, axis=-1)

        for one_step in range(len(encodings_part)):
            assert global_index < len(encodings), f'global_index < len(encodings)'
            if len(encodings) == 1:
                # 不超过一个128 也不一定有128行
                last_mask = encodings[global_index][3][0]
                global_indexs[:] = indexs[one_step, last_mask]
                global_values[:] = values[one_step, last_mask]
            else:
                this_start = global_index * args.max_seq_len - 20 * global_index + 10
                if global_index == 0:
                    global_indexs[0: args.max_seq_len - 10] = indexs[one_step, :-10]
                    global_values[0: args.max_seq_len - 10] = values[one_step, :-10]
                    pass
                elif global_index == len(encodings) - 1:
                    # 全局最后一个 不一定有128行
                    last_mask = encodings[global_index][3][0]
                    global_indexs[this_start:] = indexs[one_step, last_mask][10:]
                    global_values[this_start:] = values[one_step, last_mask][10:]
                else:
                    global_indexs[this_start: this_start + args.max_seq_len - 20] = indexs[one_step, 10:-10]
                    global_values[this_start: this_start + args.max_seq_len - 20] = values[one_step, 10:-10]
            global_index += 1

    entities = get_entity_bieos([id2label[i] for i in global_indexs])
    for index in range(len(entities)):
        start = entities[index][1] - 1
        end = entities[index][2] + 1
        tags = global_values[start:end]
        confidence = np.mean(np.array(tags))
        entities[index] = entities[index] + (round(float(confidence), 6),)

    return entities


class Dict2Class:
    def __init__(self, **entries):
        self.__dict__.update(entries)


torch_env()
model_name = './checkpoints/财务附注定位V2-chinese-roberta-small-wwm-cluecorpussmall-2024-06-26'

args_path = os.path.join(model_name, 'args.json')
labels_path = os.path.join(model_name, 'labels.json')

port = 10086
with open(args_path, "r", encoding="utf-8") as fp:
    tmp_args = json.load(fp)
with open(labels_path, 'r', encoding='utf-8') as f:
    label_list = json.load(f)
id2label = {k: v for k, v in enumerate(label_list)}
label2id = {v: k for k, v in enumerate(label_list)}
args = Dict2Class(**tmp_args)
tokenizer = BertTokenizerFast.from_pretrained(args.bert_dir)



onnx_path = os.path.join(model_name, 'model.onnx')
onnx_path = Path(onnx_path)

try:
    print('load onnx model')
    onnx_path = os.path.join(model_name, 'model.onnx')
    onnx_path = Path(onnx_path)

    print(onnxruntime.get_device())
    onnx_model = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'], provider_options=[{'device_id': int(args.gpu_ids)}])
    print('load onnx model success')
except Exception as e:
    print(str(e))
    print(traceback.format_exc())

app = Flask(__name__)


@app.route('/prediction', methods=['POST'])
def prediction():
    # noinspection PyBroadException
    try:
        msgs = request.get_data()
        # msgs = request.get_json("content")
        msgs = msgs.decode('utf-8')
        msgs = json.loads(msgs)
        assert type(msgs) == list, '输入应为list of str'
        # print(msg)
        # 是否需对句子数量限制 false 是否对单句长度限制 false
        encodings = encode(msgs)
        results = decode(encodings, len(msgs))
        for _ in trange(10000):
            results = decode(encodings, len(msgs))
        res = json.dumps(results, ensure_ascii=False)
        # torch.cuda.empty_cache()
        return res
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, threaded=False, debug=False)
    # server = pywsgi.WSGIServer(('0.0.0.0', port), app)
    print("Starting server in python...")
    print('Service Address : http://' + get_ip_config() + ':' + str(port))
    # server.serve_forever()
    print("done!")
    # app.run(host=hostname, port=port, debug=debug)  注释以前的代码
    # manager.run()  # 非开发者模式
