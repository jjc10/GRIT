import sys
import os
# THIS IS NEEDED SO IMPORTS WORK PROPERLY. FIX PERMANENTLY BY ADDRESSING IMPORTS WITHOUT RELYING ON INTELLIJ MODIFYING THE PATH
cwd = os.getcwd()
root_abs_path_index = cwd.split("/").index("GRIT")
project_root = "/".join(os.getcwd().split("/")[:root_abs_path_index + 1])
sys.path.insert(0, project_root)

sys.path.insert(0, f'{project_root}/dynn')

import torch
import logging
import torch.optim as optim
import mlflow

from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                            # set_agg_dir,
                                            set_cfg, load_cfg,
                                            makedirs_rm_exist)
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.loader import create_loader
from torch_geometric import seed_everything
import argparse
from torch_geometric.graphgym.model_builder import create_model
from grit.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from dynn_wrapper import DynnWrapper
from dynn.training_helper import LearningHelper, TrainingPhase
from main import custom_set_out_dir, new_optimizer_config, new_scheduler_config
from dynn.log_helper import setup_mlflow
from grit.utils import fix_the_seed
from torch_geometric.graphgym.checkpoint import get_ckpt_dir, get_ckpt_epoch, get_ckpt_path

fix_the_seed(42)

def set_from_validation(training_helper, val_metrics_dict, freeze_classifier_with_val=False, alpha_conf = 0.04):

    # we fix the 1/0 ratios of gate tasks based on the optimal percent exit in the validation sets

    exit_count_optimal_gate = val_metrics_dict['exit_count_optimal_gate'] # ({0: 0, 1: 0, 2: 0, 3: 0, 4: 6, 5: 72}, 128)
    total = exit_count_optimal_gate[1]
    pos_weights = []
    pos_weights_previous = []
    for gate, count in exit_count_optimal_gate[0].items():
        count = max(count, 0.1)
        pos_weight = (total-count) / count # #0/#1
        pos_weight = min(pos_weight, 5) # clip for stability
        pos_weights.append(pos_weight)
    training_helper.gate_training_helper.set_ratios(pos_weights)

def parse_args():
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=True,
                        help='The configuration file path.')
    parser.add_argument('--ce_ic_tradeoff', default=0.1, type=float,
                        help='CE IC tradeoff, the higher the earlier we exit')
    parser.add_argument('--arch', default='jeidnn', type=str,
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


args, cfg = parse_args_and_cgf()
loaders = create_loader()
model = create_model()

model = init_model_from_pretrained(
    model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
    reset_prediction_head=False, freeze_final_head=True
)
NUM_CLASSES = 10
exit_positions = [i for i in range(cfg.gt.layers - 1)]
dynn = DynnWrapper(grit_transformer=model.model, head_dim_in = cfg.gt.dim_hidden,
                   head_dim_out = NUM_CLASSES, ce_ic_tradeoff=args.ce_ic_tradeoff) # graph gym module wraps around the model.
dynn.set_intermediate_heads(exit_positions)
dynn.set_learnable_gates(exit_positions)
logging.info(model)
logging.info(cfg)
dynn = dynn.to(cfg.accelerator) # move to gpu
# logging.info('Num parameters: %s', cfg.params)
# Start training
experiment_name = 'jeidnn'

setup_mlflow("CIFAR LARGE", args, experiment_name)
trainable_params = list(filter(lambda x: x.requires_grad, list(dynn.parameters())))
named_trainable_params = list(map(lambda x: x[0], filter(lambda x: x[1].requires_grad, list(dynn.named_parameters()))))
print(f'Trainable params after augmenting models: {named_trainable_params}')
optimizer = create_optimizer(iter(trainable_params),
                             new_optimizer_config(cfg))
scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))

learning_helper = LearningHelper(dynn, optimizer, scheduler, cfg)

bilevel_batch_count = 200
best_acc = 0


for g in optimizer.param_groups: # manually set the start learning rate
    g['lr'] = cfg.optim.base_lr
for warmup_epoch in range(cfg.optim.num_warmup_epochs):
    learning_helper.train_single_epoch(loaders[0], warmup_epoch, TrainingPhase.WARMUP)
    val_metrics_dict, best_acc, _ = learning_helper.evaluate(best_acc, loaders[1], epoch=warmup_epoch, mode='val', arch=args.arch, experiment_name=experiment_name)
training_phase = TrainingPhase.CLASSIFIER
for bilevel_epoch in range(warmup_epoch, cfg.optim.max_epoch):
    learning_helper.train_single_epoch(loaders[0], bilevel_epoch, training_phase)
    val_metrics_dict, new_best_acc, _ = learning_helper.evaluate(best_acc, loaders[1], epoch=bilevel_epoch, mode='val', arch=args.arch, experiment_name=experiment_name)
    store_results = new_best_acc > best_acc
    learning_helper.evaluate(best_acc, loaders[2], epoch=bilevel_epoch, mode='test', arch=args.arch, experiment_name=experiment_name, store_results=True)
    set_from_validation(learning_helper, val_metrics_dict)


