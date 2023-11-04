import datetime
import os
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
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.model_builder import create_model
from grit.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from dynn_wrapper import DynnWrapper
from training_helper import LearningHelper
from main import custom_set_out_dir, new_optimizer_config, new_scheduler_config
from torch_geometric.graphgym.checkpoint import get_ckpt_dir, get_ckpt_epoch, get_ckpt_path
def setup_mlflow(run_name: str, cfg, experiment_name):
    print(run_name)
    project = experiment_name
    mlruns_path = get_abs_path(["mlruns"])
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(project)
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(cfg)

def get_abs_path(paths_strings):
    subpath = "/".join(paths_strings)
    src_abs_path = get_path_to_project_root()
    return f'{src_abs_path}/{subpath}/'

def get_path_to_project_root():
    cwd = os.getcwd()
    root_abs_path_index = cwd.split("/").index("GRIT")
    return "/".join(os.getcwd().split("/")[:root_abs_path_index + 1])

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

init_model_from_pretrained(
    model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
    False
)
NUM_CLASSES = 10
dynn = DynnWrapper(grit_transformer=model.model, head_dim_in = cfg.gt.dim_hidden, head_dim_out = NUM_CLASSES) # graph gym module wraps around the model.
dynn.set_intermediate_heads([i for i in range(cfg.gt.layers - 1)])
logging.info(model)
logging.info(cfg)
dynn = dynn.to(cfg.accelerator) # move to gpu
# logging.info('Num parameters: %s', cfg.params)
# Start training
setup_mlflow("warmup", {"a": 1}, "train_dynn")
optimizer = create_optimizer(dynn.parameters(),
                             new_optimizer_config(cfg))

learning_helper = LearningHelper(dynn, optimizer)
scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
for idx, batch in enumerate(loaders[0]):
    batch = batch.to(cfg.accelerator)
    loss, things_of_interest = learning_helper.get_warmup_loss(batch, batch.y)
    losses = things_of_interest['losses']
    mlflow_dict = {f'loss_{idx}': float(v) for idx, v in enumerate(losses)}
    mlflow.log_metrics(mlflow_dict, idx)
    loss.backward()
    learning_helper.optimizer.step()


