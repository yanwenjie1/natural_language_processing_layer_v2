# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/4
@Time    : 16:31
@File    : test.py
@Function: XX
@Other: XX
"""
import json
import requests
from tqdm import trange


if __name__ == '__main__':
    with open('财务附注测试用例.txt', 'r', encoding='utf-8') as f:
        texts = f.readlines()

    texts = texts + texts
    # texts = texts + texts
    # texts = texts + texts
    input_str = json.dumps(texts, ensure_ascii=False)

    results = requests.post(url='http://10.17.107.65:10086/prediction', data=input_str.encode("utf-8")).text

    # print(results)
    # results = server_test(json.dumps([''], ensure_ascii=False))
    print(results)
    # results = json.loads(results)
    # print(results)

    # print(results_old)
    # print(results_new)

    for _ in trange(500):
        results = requests.post(url='http://10.17.107.65:10086/prediction', data=input_str.encode("utf-8")).text
    #
    # for _ in trange(50):
    #     results_new = requests.post(url='http://10.17.107.65:10089/prediction', data=input_str.encode("utf-8")).text
    #