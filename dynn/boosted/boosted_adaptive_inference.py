from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
import argparse
import numpy as np
import torch
import torch.nn as nn
import os
from torch import nn, optim
import mlflow
from grit.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
import logging
import torch.backends.cudnn as cudnn
from dynn.boosted_wrapper import BoostedWrapper
import math
import time
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.loader import create_loader
from main import custom_set_out_dir


from dynn.log_helper import setup_mlflow
from dynn.gates.gate import GateType
from torch_geometric.graphgym.config import (cfg, dump_cfg,
    # set_agg_dir,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
# from models.op_counter import measure_model_and_assign_cost_per_exit

import pickle as pk

class CustomizedOpen():
    def __init__(self, path, mode): 
        self.path = path
        self.mode = mode

    def __enter__(self):
        self.f = open(self.path, self.mode)
        return self.f

    def __exit__(self, type, value, traceback):
        self.f.close()


    
def stolen_calibrate(logits, targets, temp=None):
        

        if temp is None: # we compute the temp on this data
            # Stolen from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py#L78
            nll_criterion = nn.CrossEntropyLoss().cuda()
            #ece_criterion = _ECELoss().cuda()
            before_temperature_nll = nll_criterion(logits, targets.long()).item()
        # before_temperature_ece = ece_criterion(logits, labels).item()
           
            temp = nn.Parameter(torch.ones(1) * 1.5)
            # Next: optimize the temperature w.r.t. NLL
            optimizer = optim.LBFGS([temp], lr=0.01, max_iter=50)

            def eval():
                optimizer.zero_grad()
                loss = nll_criterion(logits/temp, targets.long())
                loss.backward()
                return loss
            optimizer.step(eval)

            # Calculate NLL and ECE after temperature scaling
            after_temperature_nll = nll_criterion(logits/temp, targets.long()).item()
           
            return temp.detach().cpu().numpy(), logits/temp
        else: # we calibrate the logits with the given temp
            return logits/temp

def dynamic_evaluate(model, test_loader, val_loader, args, cfg):
    tester = Tester(model, args)
    flops = model.cost_per_exit()

    val_pred, val_target = tester.calc_logit(val_loader)
    test_pred, test_target = tester.calc_logit(test_loader)


    acc_val_last = -1
    acc_test_last = -1

    save_path = os.path.join(cfg.out_dir, 'dynamic_mnist.txt')

    with CustomizedOpen(save_path, 'w') as fout:
        # for p in range(1, 100):
        for p in range(1, 40):
            print("*********************")
            _p = torch.FloatTensor(1).fill_(p * 1.0 / 20)
            n_blocks = len(model.intermediate_heads) + 1
            probs = torch.exp(torch.log(_p) * torch.range(1, n_blocks))
            probs /= probs.sum()
            acc_val, _, T = tester.dynamic_eval_find_threshold(val_pred, val_target, probs, flops)
            acc_test, exp_flops = tester.dynamic_eval_with_threshold(test_pred, test_target, flops, T)
            print('valid acc: {:.3f}, test acc: {:.3f}, test flops: {:.2f}M'.format(acc_val, acc_test, exp_flops))
            fout.write('{} {}\n'.format(exp_flops.item(), acc_test))




class Tester(object):
    def __init__(self, model, args=None):
        self.args = args
        self.model = model
        self.softmax = nn.Softmax(dim=1).cuda()

    def calc_logit(self, dataloader):
        self.model.eval()
        n_stage = len(self.model.intermediate_heads) + 1
        logits = [[] for _ in range(n_stage)]
        targets = []
        for batch_idx, batch in enumerate(dataloader):
            input = batch
            target = batch.y
            input = input.cuda()
            target = target.cuda()
            target = target.cpu()
            targets.append(target)
            with torch.no_grad():
                # input_var = torch.autograd.Variable(input)
                input_var = input
                # if True:
                #     foo = self.model.forward(input_var)
                #     intermediate_logits = list(map(lambda x: x[0], intermediate_logits))
                #     intermediate_logits.append(last_head_out)
                #     output = intermediate_logits
                # else:
                output = self.model.forward(input_var)
                if not isinstance(output, list):
                    output = [output]
                for b in range(n_stage):
                    _t = self.softmax(output[b])
                    _t = _t.cpu()
                    logits[b].append(_t)

            if batch_idx % 50 == 0:
                print('Generate Logit: [{0}/{1}]'.format(batch_idx, len(dataloader)))

        for b in range(n_stage):
            logits[b] = torch.cat(logits[b], dim=0)

        size = (n_stage, logits[0].size(0), logits[0].size(1))
        ts_logits = torch.Tensor().resize_(size).zero_()
        for b in range(n_stage):
            ts_logits[b].copy_(logits[b])

        targets = torch.cat(targets, dim=0)
        ts_targets = torch.Tensor().resize_(size[1]).copy_(targets)

        return ts_logits, ts_targets

    def dynamic_eval_find_threshold(self, logits, targets, p, flops):
        """
            logits: m * n * c
            m: Stages
            n: Samples
            c: Classes
        """
        n_stage, n_sample, c = logits.size()

        max_preds, argmax_preds = logits.max(dim=2, keepdim=False)

        _, sorted_idx = max_preds.sort(dim=1, descending=True)

        filtered = torch.zeros(n_sample)
        T = torch.Tensor(n_stage).fill_(1e8)

        for k in range(n_stage - 1):
            acc, count = 0.0, 0
            out_n = math.floor(n_sample * p[k])
            for i in range(n_sample):
                ori_idx = sorted_idx[k][i]
                if filtered[ori_idx] == 0:
                    count += 1
                    if count == out_n:
                        T[k] = max_preds[k][ori_idx]
                        break
            filtered.add_(max_preds[k].ge(T[k]).type_as(filtered))

        T[n_stage -1] = -1e8 # accept all of the samples at the last stage

        acc_rec, exp = torch.zeros(n_stage), torch.zeros(n_stage)
        acc, expected_flops = 0, 0
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_stage):
                if max_preds[k][i].item() >= T[k]: # force the sample to exit at k
                    if int(gold_label.item()) == int(argmax_preds[k][i].item()):
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break
        acc_all = 0
        for k in range(n_stage):
            _t = 1.0 * exp[k] / n_sample
            expected_flops += _t * flops[k]
            acc_all += acc_rec[k]

        return acc * 100.0 / n_sample, expected_flops, T

    def dynamic_eval_with_threshold(self, logits, targets, flops, T):
        n_stage, n_sample, _ = logits.size()
        max_preds, argmax_preds = logits.max(dim=2, keepdim=False) # take the max logits as confidence

        acc_rec, exp = torch.zeros(n_stage), torch.zeros(n_stage)
        acc, expected_flops = 0, 0
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_stage):
                if max_preds[k][i].item() >= T[k]: # force to exit at k
                    _g = int(gold_label.item())
                    _pred = int(argmax_preds[k][i].item())
                    if _g == _pred:
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break
        acc_all, sample_all = 0, 0
        for k in range(n_stage):
            _t = exp[k] * 1.0 / n_sample
            sample_all += exp[k]
            expected_flops += _t * flops[k]
            acc_all += acc_rec[k]

        return acc * 100.0 / n_sample, expected_flops



def load_model_from_checkpoint(checkpoint_path, device, num_classes):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model = create_model()
    exit_positions = [i for i in range(cfg.gt.layers - 1)]
    wrapper = BoostedWrapper(grit_transformer=model.model, head_dim_in = cfg.gt.dim_hidden,
                             head_dim_out = num_classes)
    wrapper.set_intermediate_heads(exit_positions)
    logging.info(model)
    logging.info(cfg)

    # TODO: fix this for weighted to serialize where the intermediate head positions were
    # net.set_intermediate_heads(checkpoint['intermediate_head_positions'])


    # if 'state_dict' in checkpoint.keys():
    #     wrapper.load_state_dict(checkpoint['state_dict'], strict=False)
    # else:
    #
    wrapper.load_state_dict(checkpoint['net'], strict=False)
    wrapper = wrapper.to(cfg.accelerator) # move to gpu
    return wrapper

def main(args, cfg, val_loader, test_loader):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 10
    dataset = cfg.dataset.name.lower()
    if dataset == 'mnist':
        num_classes = 10

    else:
        raise 'Unsupported dataset'

     # LOAD MODEL
    checkpoint_path = '/home/joud/code/relu_analysis/GRIT/dynn_results/mnist-GRIT-RRWP-LARGE/boosted/mnist-GRIT-boosted/boosted/ckpt_89.84.pth'
    print(f"Loading model with checkpoint path {checkpoint_path}")

    net = load_model_from_checkpoint(checkpoint_path, device, num_classes)
    # measure_model_and_assign_cost_per_exit(net.module, IMG_SIZE, IMG_SIZE, NUM_CLASSES)

    net = net.to(device)
    dynamic_evaluate(net, test_loader, val_loader, args, cfg)

def parse_args():
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=True,
                        help='The configuration file path.')
    parser.add_argument('--arch', default='boosted', type=str,
                        help='architecture')
    parser.add_argument('--repeat', type=int, default=1,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')
    return parser.parse_args()

def parse_args_and_cgf():
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    # ----- note: allow to change config -----------
    cfg.set_new_allowed(True)
    cfg.work_dir = os.getcwd()
    # -----------------------------
    load_cfg(cfg, args)
    cfg.cfg_file = args.cfg_file
    # Load cmd line args

    # -----------------------------
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)
    return args, cfg





if __name__ == '__main__':
    args, cfg = parse_args_and_cgf()
    loaders = create_loader()
    main(args, cfg, loaders[1], loaders[2])
