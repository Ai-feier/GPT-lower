# -*- coding: utf-8 -*-
import os
import json

import torch
from torch.utils.data import DataLoader

from dataset_wb import *

SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]"]




def build_dist_loaders(args, tokenizer, logger):
    logger.info("Build train and validation dataloaders")

    train_dataset = LCCCdiskDataset(tokenizer, data_path=args.train_path)
    valid_dataset = LCCCdiskDataset(tokenizer, data_path=args.valid_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset,
                              collate_fn=train_dataset.collate,
                              pin_memory=(args.device == "cuda"),
                              num_workers=args.num_workers,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset,
                              collate_fn=valid_dataset.collate,
                              pin_memory=(args.device == "cuda"),
                              num_workers=args.num_workers,
                              sampler=valid_sampler,
                              batch_size=args.valid_batch_size,
                              shuffle=False)
    return train_loader, valid_loader, train_sampler, valid_sampler


