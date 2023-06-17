from torch.cuda import amp

from dataset_wb import *
from inputter import *
import logging
from pprint import pformat
from transformers import BertTokenizer, GPT2LMHeadModel
from argparse import ArgumentParser
import numpy as np
import random
import math
import os

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear, LRScheduler
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import (OpenAIGPTLMHeadModel, OpenAIGPTConfig, GPT2Config,
                          WEIGHTS_NAME, CONFIG_NAME, AdamW, BertTokenizer)
from pytorch_transformers import GPT2LMHeadModel

def main():
    '''
        更新原始模型，保存更新了embedding层的模型
    Returns:

    '''
    parser = ArgumentParser()
    parser.add_argument('--gpt2', action='store_true', help="use gpt2")
    parser.add_argument("--model_checkpoint", type=str, default="./model/bert/", help="Path or URL of the model")
    parser.add_argument("--from_step", type=int, default=-1, help="Init learning rate from this step")
    parser.add_argument('--pretrained', action='store_true', help="If False train from scratch")
    parser.add_argument("--data_path", type=str, default="",
                        help="Path or url of the dataset. ")
    parser.add_argument("--train_path", type=str, default="./dataset/LCCC-base_test.json",
                        help="Path of the train dataset for dist dataset. ")
    parser.add_argument("--valid_path", type=str, default="./dataset/LCCC-base_valid.json",
                        help="Path of the valid dataset for dist dataset. ")
    parser.add_argument("--test_path", type=str, default="./dataset/LCCC-base_test.json",
                        help="Path or url of the dataset cache")
    parser.add_argument('--log_file', '-log_file', type=str, default="", help="Output logs to a file under this path")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of subprocesses for data loading")
    parser.add_argument("--n_epochs", type=int, default=70, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=2, help="Batch size for validation")
    parser.add_argument("--max_history", type=int, default=15, help="Number of previous exchanges to keep in history")
    parser.add_argument("--scheduler", type=str, default="noam", choices=['noam', 'linear'], help="method of optim")
    parser.add_argument("--n_emd", type=int, default=768, help="Number of n_emd in config file (for noam)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Warm up steps")
    parser.add_argument("--valid_steps", type=int, default=5000, help="Perfom validation every X steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="",
                        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")

    args = parser.parse_args()

    # 打印初始pytorch_model的大小
    print("Size pytorch_model: ", os.path.getsize('./model/bert/pytorch_model.bin'))

    # model的初始化
    model = GPT2LMHeadModel.from_pretrained(args.model_checkpoint)

    # 由于词表有所更改，需要重新更换模型的嵌入层大小
    embedding = nn.Embedding(13088, 768)

    # 加载GPT-2模型
    model = GPT2LMHeadModel.from_pretrained(args.model_checkpoint)
    # 获取其lm_head权重
    pretrained_lm_head_weight = model.lm_head.weight.data

    # 替换原始权重,加载我的embedding层
    model.transformer.wte = embedding
    model.config.vocab_size = 13088

    # 定义新的lm_head权重形状
    new_lm_head_weight_shape = (13088, 768)

    # 将预训练模型的lm_head权重转换为新的形状
    new_lm_head_weight = pretrained_lm_head_weight[:new_lm_head_weight_shape[0], :]
    model.lm_head.weight.data = new_lm_head_weight

    model.to(args.device)
    model.save_pretrained('./model/emb_gpt2/')
    print('Save Model OK!!!')
    # 打印初始gpt2-130088模型的大小
    print("Size new model: ", os.path.getsize('model/emb_gpt2/pytorch_model.bin'))


if __name__ == "__main__":
    main()