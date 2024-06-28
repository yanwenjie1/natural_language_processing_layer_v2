# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2024/5/13
@Time    : 10:04
@File    : onnx_convert.py
@Function: XX
@Other: XX
"""
import torch
import os
from pathlib import Path
from functions import load_model_and_parallel
from server_models import LayerLabeling
import logging


def onnxconvert(args, log:logging.Logger):
    """
    训练好的模型转换为 onnx 模型
    """
    log.info('start onnxconvert')
    model = load_model_and_parallel(LayerLabeling(args), args.device, os.path.join(args.save_path, 'model_best.pt'))
    
    max_batch = max(8, args.batch_size)
    
    model.eval()
    model.half()
    log.info('end model load')
    input_names=["token_ids", "attention_masks", "token_type_ids"]
    output_names = ["outputs"]
    symbolic_names = {0: 'batch_size'}
    dev_inputs = (
                torch.ones(max_batch, args.max_seq_len, args.max_word_len, dtype=torch.int32, device=args.device),
                torch.ones(max_batch, args.max_seq_len, args.max_word_len, dtype=torch.int32, device=args.device),
                torch.ones(max_batch, args.max_seq_len, args.max_word_len, dtype=torch.int32, device=args.device))

    onnx_path = Path(os.path.join(args.save_path, 'model.onnx'))

    with torch.no_grad():
        torch.onnx.export(model, 
                        dev_inputs, 
                        onnx_path, 
                        verbose=False, 
                        do_constant_folding=True,
                        input_names=input_names, 
                        output_names=output_names, 
                        dynamic_axes={
                            'token_ids': symbolic_names,
                            'attention_masks': symbolic_names,
                            'token_type_ids': symbolic_names},
                        opset_version=17)
    log.info('end torch.onnx.export')
    pass
