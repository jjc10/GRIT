import sys
import os
import datetime
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
from boosted_wrapper import BoostedWrapper
from boosted_training_helper import get_boosted_loss, train_boosted, test_boosted
from main import custom_set_out_dir, new_optimizer_config, new_scheduler_config
from dynn.log_helper import setup_mlflow
from grit.utils import fix_the_seed, save_model_checkpoint
from torch_geometric.graphgym.checkpoint import get_ckpt_dir, get_ckpt_epoch, get_ckpt_path

fix_the_seed(42)

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


args, cfg = parse_args_and_cgf()
loaders = create_loader()
model = create_model()

model = init_model_from_pretrained(
    model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
    reset_prediction_head=False, freeze_final_head=True
)
NUM_CLASSES = 10
exit_positions = [i for i in range(cfg.gt.layers - 1)]
wrapper = BoostedWrapper(grit_transformer=model.model, head_dim_in = cfg.gt.dim_hidden,
                         head_dim_out = NUM_CLASSES)
wrapper.set_intermediate_heads(exit_positions)
logging.info(model)
logging.info(cfg)
dynn = wrapper.to(cfg.accelerator) # move to gpu
# logging.info('Num parameters: %s', cfg.params)
# Start training
experiment_name = 'boosted'

setup_mlflow("MNIST LARGE", args, experiment_name)
trainable_params = list(filter(lambda x: x.requires_grad, list(dynn.parameters())))
named_trainable_params = list(map(lambda x: x[0], filter(lambda x: x[1].requires_grad, list(dynn.named_parameters()))))
print(f'Trainable params after augmenting models: {named_trainable_params}')
optimizer = create_optimizer(iter(trainable_params),
                             new_optimizer_config(cfg))
scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))



bilevel_batch_count = 200
best_val_acc = 0


for g in optimizer.param_groups: # manually set the start learning rate
    g['lr'] = cfg.optim.base_lr

start_time = datetime.datetime.now()
ckpt_dir_suffix = f'{start_time.month}-{start_time.day}-{start_time.hour}-{start_time.minute}'
for epoch in range(0, cfg.optim.max_epoch + cfg.optim.num_warmup_epochs):
    train_boosted(wrapper, cfg.accelerator, loaders[0], optimizer, epoch)
    accs = test_boosted(wrapper, loaders[1], cfg.accelerator)
    val_acc_last_inter_head = accs[-2]
    if val_acc_last_inter_head > best_val_acc:
        print(f"Increase in validation accuracy to {val_acc_last_inter_head} of last trainable head of boosted, serializing model")
        best_val_acc = val_acc_last_inter_head
        save_model_checkpoint(wrapper, best_val_acc, epoch, cfg, f'boosted')    # stored_metrics_test = test(epoch)
    scheduler.step()


