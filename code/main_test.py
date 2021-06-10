import torch
import os

import data
import model
import loss
import utility
from option import args
import trainer
import copy
import template
from torchsummaryX import summary

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    args.test_only = True
    args.data_test = 'Set5'  # specify dataset for test.
#    args.save_results = True
    loader = data.Data(args)
    ckpt = utility.checkpoint(args)
    model = model.Model(args, ckpt)
    loss = loss.Loss(args, ckpt) if not args.test_only else None
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    t = trainer.Trainer(args, loader, model, loss, ckpt)
    t.test()

    ckpt.done()


