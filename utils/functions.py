# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/6
@Time    : 10:11
@File    : functions.py
@Function: XX
@Other: XX
"""
import os
import torch
import random
import numpy as np
import logging
import json
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from rich.table import Table
from rich.align import Align
from rich.console import Console


def set_seed(seed=2022):
    """
    设置随机数种子，保证实验可重现
    :param seed:
    :return:
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # 固定随机种子 CPU
    torch.cuda.manual_seed(seed)  # 固定随机种子 当前GPU
    torch.cuda.manual_seed_all(seed)  # 固定随机种子 所有GPU
    np.random.seed(seed)  # 保证后续使用random时 产生固定的随机数
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 固定网络结构
    torch.backends.cudnn.benchmark = False  # GPU和网络结构固定时 可以为True 自动寻找更优
    torch.backends.cudnn.enabled = False


def load_model_and_parallel(model, device, ckpt_path=None, strict=True):
    # set to device to the first cuda
    if ckpt_path is not None:
        print('Load ckpt from {}'.format(ckpt_path))
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')), strict=strict)
    model = model.to(device)
    return model


# def load_model_and_parallel(model, gpu_ids, ckpt_path=None, strict=True):
#     """
#     加载模型 & 放置到 GPU 中（单卡 / 多卡）
#     :param model: 实例化的模型对象
#     :param gpu_ids: GPU的ID
#     :param ckpt_path: 模型加载路径
#     :param strict: 是否严格加载
#     :return:
#     """
#     gpu_ids = gpu_ids.split(',')
#     # set to device to the first cuda
#     device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
#     # 用strict=False进行加载模型，则 能塞则塞 不能塞则丢。
#     # strict=True时一旦有key不匹配则出错，如果设置strict=False，则直接忽略不匹配的key，对于匹配的key则进行正常的赋值。
#     if ckpt_path is not None:
#         print('Load ckpt from {}'.format(ckpt_path))
#         model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')), strict=strict)

#     if len(gpu_ids) > 1:
#         print('Use multi gpus in: {}'.format(gpu_ids))
#         gpu_ids = [int(x) for x in gpu_ids]
#         model = torch.nn.DataParallel(model, device_ids=gpu_ids)
#     else:
#         print('Use single gpu in: {}'.format(gpu_ids))
#         model = model.to(device)

#     return model, device


def build_optimizer_and_scheduler(args, model, t_total):
    """
    不同的模块使用不同的学习率
    :param args:
    :param model:
    :param t_total:
    :return:
    """
    # hasattr 判断对象是否包含对应的属性
    module = (
        model.module if hasattr(model, "module") else model
    )

    # 差分学习率
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    crf_param_optimizer = []
    other_param_optimizer = []

    for name, para in model_param:
        space = name.split('.')
        # print(name)
        if space[0] == 'bert_module':
            bert_param_optimizer.append((name, para))
        elif space[0] == 'crf':
            crf_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    optimizer_grouped_parameters = [
        # bert other module
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.lr},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.lr},

        # crf模块
        {"params": [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.crf_lr},
        {"params": [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.other_lr},

        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.other_lr},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.other_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_proportion * t_total), num_training_steps=t_total
    )
    # opt = SWA(optimizer, swa_start=10, swa_freq=5, swa_lr=0.05)  #  后面尝试swa的效果

    return optimizer, scheduler


# def save_model(args, model, global_step, log):
#     """
#     保存验证集效果最好的那个模型
#     :param log:
#     :param args:
#     :param model:
#     :param global_step:
#     :return:
#     """
#     # take care of model distributed / parallel training  小心分布式训练or并行训练
#     model_to_save = (
#         model.module if hasattr(model, "module") else model
#     )
#     # log.info('Saving model checkpoint to {}'.format(output_dir))
#     # torch.save(model_to_save.state_dict(), os.path.join(args.save_path, 'model_{}.pt'.format(global_step)))
#     torch.save(model_to_save.state_dict(), os.path.join(args.save_path, 'model_best.pt'))


def save_model(args, model, log:logging.Logger):
    """
    :param log:
    :param args:
    :param model:
    :param global_step:
    :return:
    """
    # take care of model distributed / parallel training
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    log.info('Saving model checkpoint to {}'.format(args.save_path))
    # torch.save(model_to_save.state_dict(), os.path.join(args.save_path, 'model_{}.pt'.format(global_step)))
    torch.save(model_to_save.state_dict(), os.path.join(args.save_path, 'model_best.pt'))


def set_logger(log_path):
    """
    配置log
    :param log_path:s
    :return:
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 由于每调用一次set_logger函数，就会创建一个handler，会造成重复打印的问题，因此需要判断root logger中是否已有该handler
    if not any(handler.__class__ == logging.FileHandler for handler in logger.handlers):
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not any(handler.__class__ == logging.StreamHandler for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def reset_console(args):
    """
    重置终端，便于打印log信息。
    """
    console = Console()
    table = Table(show_footer=False)
    table.title = ("[bold not italic]:robot:[/] Config Parameters")
    table.add_column("key", no_wrap=True)
    table.add_column("value", no_wrap=True)

    for arg in vars(args):
        table.add_row(arg, str(getattr(args, arg)))

    table.caption = "You can change config in [b not dim]Source Code[/]"
    table.columns[0].style = "bright_red"
    table.columns[0].header_style = "bold bright_red"
    table.columns[1].style = "bright_green"
    table.columns[1].header_style = "bold bright_green"
    table_centered = Align.center(table)
    console.print(table_centered)


def save_json(data_dir, data, desc):
    """保存数据为json"""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    with open(os.path.join(data_dir, '{}.json'.format(desc)), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_entity_bieos(labels):
    """
    输入标签的List 输出每一个实体
    :param labels:
    :return:list of (label, start, end)
    """
    entities = []
    entity = []
    for i in range(len(labels)):
        if labels[i] == "O":
            entity = []
        elif labels[i][0] == "B":
            entity = [labels[i][2:], i]
        elif labels[i][0] == "I" and len(entity) == 2 and labels[i][2:] == entity[0]:
            continue
        elif labels[i][0] == "E" and len(entity) == 2 and labels[i][2:] == entity[0]:
            entity.append(i + 1)
            entities.append(entity[:])
            entity = []
        elif labels[i][0] == "S" and labels[i] != "SEP":
            entity = [labels[i][2:], i, i + 1]
            entities.append(entity[:])
            entity = []
        else:
            entity = []
    return [tuple(i) for i in entities]


def to_numpy(tensor:torch.Tensor):
    tensor = tensor.to(torch.int32)
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
