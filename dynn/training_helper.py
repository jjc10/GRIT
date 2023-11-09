import torch
import copy
from torch import nn
import numpy as np
import mlflow
from grit.utils import progress_bar, split_dataloader_in_n, aggregate_dicts
from dynn.log_helper import process_things, aggregate_metrics, log_aggregate_metrics_mlflow, get_path_to_project_root
from dynn.classifier_training_helper import ClassifierTrainingHelper, LossContributionMode
from dynn.gate_training_helper import GateTrainingHelper, GateObjective
from enum import Enum
import os
import pickle as pk


criterion = nn.CrossEntropyLoss()
class TrainingPhase(Enum):
    CLASSIFIER = 1
    GATE = 2
    WARMUP = 3

class LearningHelper:
    def __init__(self, net, optimizer, scheduler, cfg) -> None:
        self.net = net
        self.optimizer = optimizer
        self.cfg = cfg
        self.device = self.cfg.accelerator
        self.scheduler = scheduler
        self._init_classifier_training_helper()
        self._init_gate_training_helper()

    def _init_classifier_training_helper(self) -> None:
        self.loss_contribution_mode = LossContributionMode.BOOSTED
        self.classifier_training_helper = ClassifierTrainingHelper(self.net,
                                                                   self.loss_contribution_mode,
                                                                   self.net.num_of_gates,
                                                                   self.device)

    def _init_gate_training_helper(self) -> None:
        self.gate_training_helper = GateTrainingHelper(self.net, GateObjective.CrossEntropy, self.net.num_of_gates, self.device)

    def get_surrogate_loss(self, inputs, targets, training_phase=None):
        if self.net.training:
            self.optimizer.zero_grad()
            if training_phase == TrainingPhase.CLASSIFIER:
                return self.classifier_training_helper.get_loss(inputs, targets)
            elif training_phase == TrainingPhase.GATE:
                return self.gate_training_helper.get_loss(inputs, targets)
        else:
            with torch.no_grad():
                copied_inputs = copy.deepcopy(inputs) # this is needed because the forward operation modifies the batch which is an object
                # passed by reference
                gate_loss, things_of_interest_gate = self.gate_training_helper.get_loss(inputs, targets)
                classifier_loss, things_of_interest = self.classifier_training_helper.get_loss(copied_inputs, targets)
                loss = (gate_loss + classifier_loss) / 2
                things_of_interest.update(things_of_interest_gate)
                return loss, things_of_interest


    def get_warmup_loss(self, batch, targets):
        criterion = nn.CrossEntropyLoss()
        self.optimizer.zero_grad()
        final_logits, intermediate_logits = self.net(batch)
        final_loss = criterion(final_logits, targets)  # the grad_fn of this loss should be None if frozen
        num_gates = len(intermediate_logits) + 1
        intermediate_losses = []
        # losses.append(loss)
        loss = 0
        for l, intermediate_logit in enumerate(intermediate_logits):
            intermediate_loss = criterion(intermediate_logit, targets)
            intermediate_losses.append(intermediate_loss)
            loss += (num_gates - l) * intermediate_loss # we scale the gradient by G-l => early gates have bigger gradient
        things_of_interest = {
            'intermediate_logits': intermediate_logits,
            'final_logits': final_logits,
            'intermediate_losses': intermediate_losses
        }
        return loss, things_of_interest

    def train_warmup(self, train_loader, val_loader):
        self.net.train()
        metrics_dict = {}
        for e in range(self.cfg.optim.num_warmup_epochs):
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(self.cfg.accelerator)

                self.optimizer.zero_grad()
                loss, things_of_interest = self.get_warmup_loss(batch, batch.y)
                losses = things_of_interest['intermediate_losses']
                mlflow_dict = {f'loss_{idx}': float(v) for idx, v in enumerate(losses)}
                progress_bar(
                    batch_idx, len(train_loader),
                    'Epoch: %d, Loss: %.3f' %
                    (e, loss))
                metrics_of_batch = process_things(things_of_interest, gates_count=self.net.num_of_gates,
                                                  targets=batch.y, batch_size=len(batch),
                                                  cost_per_exit=self.net.cost_per_exit())
                loss.backward()
                self.optimizer.step()
                metrics_dict = aggregate_metrics(metrics_of_batch, metrics_dict, gates_count=self.net.num_of_gates)

                # format the metric ready to be displayed
                log_dict = log_aggregate_metrics_mlflow(
                prefix_logger='train',
                metrics_dict=metrics_dict, gates_count=self.net.num_of_gates)
                mlflow.log_metrics(mlflow_dict,
                                   step=batch_idx +
                                        (e * len(train_loader)))
            self.scheduler.step()

    def train_single_epoch(self, train_loader, epoch, training_phase,
                           bilevel_batch_count=200):
        print('\nEpoch: %d' % epoch)
        display_count = 100
        self.net.train()
        device = self.device

        metrics_dict = {}
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch
            targets = batch.y
            inputs, targets = inputs.to(device), targets.to(device)
            losses_dict = {}
            batch_size = targets.size(0)
            if batch_idx % display_count == 0:
                total_batches = len(train_loader)
                print(f'Batch: {batch_idx}/{total_batches}, phase {training_phase}, lr: {self.optimizer.defaults["lr"]}')
            if training_phase == TrainingPhase.WARMUP:
                #  we compute the warmup loss
                loss, things_of_interest = self.get_warmup_loss(inputs, targets)
                losses = things_of_interest['intermediate_losses']
                losses_dict = {f'loss_{idx}': float(v) for idx, v in enumerate(losses)}
            else:
                if batch_idx % bilevel_batch_count == 0:

                    metrics_dict = {}
                    training_phase = switch_training_phase(training_phase)
                loss, things_of_interest = self.get_surrogate_loss(inputs, targets, training_phase)
            loss.backward()
            self.optimizer.step()

            # obtain the metrics associated with the batch
            metrics_of_batch = process_things(things_of_interest, gates_count=self.net.num_of_gates,
                                              targets=targets, batch_size=batch_size,
                                              cost_per_exit=self.net.cost_per_exit())
            metrics_of_batch['loss'] = (loss.item(), batch_size)

            # keep track of the average metrics
            metrics_dict = aggregate_metrics(metrics_of_batch, metrics_dict, gates_count=self.net.num_of_gates)

            # format the metric ready to be displayed
            log_dict = log_aggregate_metrics_mlflow(
                prefix_logger='train',
                metrics_dict=metrics_dict, gates_count=self.net.num_of_gates)

            log_dict = log_dict | losses_dict if bool(losses_dict) else log_dict
            mlflow.log_metrics(log_dict,
                               step=batch_idx +
                                    (epoch * len(train_loader)))

            #  display_progress_bar('train', training_phase, step=batch_idx, total=len(train_loader), log_dict=log_dict)

        self.scheduler.step()
        return metrics_dict

    def evaluate(self, best_acc, init_loader, epoch, mode: str, experiment_name: str, arch: str, store_results=False):
        self.net.eval()
        metrics_dict = {}
        if mode == 'test': # we should split the data and combine at the end
            loaders = split_dataloader_in_n(init_loader, n=10)
        else:
            loaders = [init_loader]
        metrics_dicts = []
        log_dicts_of_trials = {}
        average_trials_log_dict = {}
        for loader in loaders:
            for batch_idx, batch in enumerate(loader):
                inputs = batch
                targets = batch.y
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                batch_size = targets.size(0)

                loss, things_of_interest = self.get_surrogate_loss(inputs, targets)

                # obtain the metrics associated with the batch
                metrics_of_batch = process_things(things_of_interest, gates_count=self.net.num_of_gates,
                                                  targets=targets, batch_size=batch_size,
                                                  cost_per_exit=self.net.cost_per_exit())
                metrics_of_batch['loss'] = (loss.item(), batch_size)


                # keep track of the average metrics
                metrics_dict = aggregate_metrics(metrics_of_batch, metrics_dict, gates_count=self.net.num_of_gates)

                # format the metric ready to be displayed
                log_dict = log_aggregate_metrics_mlflow(
                    prefix_logger=mode,
                    metrics_dict=metrics_dict, gates_count=self.net.num_of_gates)
                #display_progress_bar(prefix_logger=prefix_logger,training_phase=TrainingPhase.CLASSIFIER, step=batch_idx, total=len(loader), log_dict=log_dict)

            metrics_dicts.append(metrics_dict)
            for k, v in log_dict.items():
                aggregate_dicts(log_dicts_of_trials, k, v)
        for k,v in log_dicts_of_trials.items():
            average_trials_log_dict[k] = np.mean(v)

        gated_acc = average_trials_log_dict[mode+'/gated_acc']
        average_trials_log_dict[mode+'/test_acc']= gated_acc
        mlflow.log_metrics(average_trials_log_dict, step=epoch)
        # Save checkpoint.
        if gated_acc > best_acc and mode == 'val':

            state = {
                'net': self.net.state_dict(),
                'acc': gated_acc,
                'epoch': epoch,
            }
            checkpoint_path = os.path.join(get_path_to_project_root(), self.cfg.out_dir)
            print(f'Saving in {checkpoint_path}...')
            this_run_checkpoint_path = os.path.join(checkpoint_path, f'checkpoint_{self.cfg.dataset.name}_{self.net.CE_IC_tradeoff}')
            if not os.path.isdir(this_run_checkpoint_path):
                os.mkdir(this_run_checkpoint_path)
            torch.save(
                state,
                os.path.join(this_run_checkpoint_path,f'ckpt_{self.net.CE_IC_tradeoff}_{gated_acc}.pth')
            )
            best_acc = gated_acc


        elif mode == 'test' and store_results:
            print('storing results....')
            pickle_res_directory = os.path.join(get_path_to_project_root(), 'pickle_results')

            if not os.path.isdir(pickle_res_directory):
                os.mkdir(pickle_res_directory)
            file_name = f'{pickle_res_directory}/{experiment_name}_{self.cfg.dataset.name}_{arch}_{str(self.net.CE_IC_tradeoff)}_results.pk'
            with open(file_name, 'wb') as file:
                pk.dump(log_dicts_of_trials, file)
        return metrics_dict, best_acc, log_dicts_of_trials

def freeze_backbone(network, excluded_submodules: list[str]):
    model_parameters = filter(lambda p: p.requires_grad, network.parameters())
    total_num_parameters = sum([np.prod(p.size()) for p in model_parameters])
    # set everything to not trainable.
    for param in network.module.parameters():
        param.requires_grad = False

    for submodule_attr_name in excluded_submodules:  # Unfreeze excluded submodules to be trained.
        for submodule in getattr(network.module, submodule_attr_name):
            for param in submodule.parameters():
                param.requires_grad = True

    trainable_parameters = filter(lambda p: p.requires_grad,
                                  network.parameters())
    num_trainable_params = sum(
        [np.prod(p.size()) for p in trainable_parameters])
    print('Successfully froze network: from {} to {} trainable params.'.format(
        total_num_parameters, num_trainable_params))

def switch_training_phase(current_phase):
    if current_phase == TrainingPhase.GATE:
        return TrainingPhase.CLASSIFIER
    elif current_phase == TrainingPhase.CLASSIFIER:
        return TrainingPhase.GATE
    elif current_phase == TrainingPhase.WARMUP:
        return TrainingPhase.GATE
