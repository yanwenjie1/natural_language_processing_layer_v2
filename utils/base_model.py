# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/5
@Time    : 13:18
@File    : base_model.py
@Function: XX
@Other: XX
"""
import os
import sys
sys.path.append(os.path.dirname(__file__))
import torch.nn as nn
from transformers import BertModel, AutoModel, AlbertModel
# from transformers import logging
# logging.set_verbosity_error()


class BaseModel(nn.Module):
    """
    基础的预训练模型
    """

    def __init__(self, bert_dir, dropout_prob=0.3, model_name=None):
        """
        利用transformers库加载预训练torch模型
        :param bert_dir: 预训练模型的路径
        :param dropout_prob: 对预训练模型的输出进行dropout
        :param model_name: 预训练模型的名字
        """
        super(BaseModel, self).__init__()
        config_path = os.path.join(bert_dir, 'config.json')
        assert os.path.exists(bert_dir) and os.path.exists(config_path), \
            'pretrained bert file does not exist'

        if 'albert' in model_name:
            self.bert_module = AlbertModel.from_pretrained(bert_dir,
                                                           output_hidden_states=False,
                                                           hidden_dropout_prob=dropout_prob)
        elif 'robert' in model_name:
            self.bert_module = BertModel.from_pretrained(bert_dir,
                                                            output_hidden_states=False,
                                                            hidden_dropout_prob=dropout_prob)
        elif 'bert' in model_name:
            # 想要接收和使用来自其他隐藏层的输出，而不仅仅是 last_hidden_state 就用True
            self.bert_module = BertModel.from_pretrained(bert_dir,
                                                         output_hidden_states=False,
                                                         hidden_dropout_prob=dropout_prob)
        else:
            self.bert_module = AutoModel.from_pretrained(bert_dir,
                                                         output_hidden_states=False,
                                                         hidden_dropout_prob=dropout_prob)

        self.base_config = self.bert_module.config

    @staticmethod
    def _init_weights(blocks, **kwargs):
        """
        参数初始化，将 Linear / Embedding / LayerNorm 与 Bert 进行一样的初始化
        """
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
