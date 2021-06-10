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

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    #os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    loader = data.Data(args)
    #template.set_template(args)
    ckpt = utility.checkpoint(args)
    model = model.Model(args, ckpt)
    loss = loss.Loss(args, ckpt) if not args.test_only else None
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    t = trainer.Trainer(args, loader, model, loss, ckpt)
    while not t.terminate():
        t.train()
        t.test()

    ckpt.done()


