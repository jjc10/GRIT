import torch
from torch import nn
import numpy as np
import mlflow
from grit.utils import progress_bar
from log_helper import process_things, aggregate_metrics, log_aggregate_metrics_mlflow
#

criterion = nn.CrossEntropyLoss()

class LearningHelper:
    def __init__(self, net, optimizer, scheduler, cfg) -> None:
        self.net = net
        self.optimizer = optimizer
        self.cfg = cfg
        self.scheduler = scheduler
        # self._init_classifier_training_helper(args, device)
        # self._init_gate_training_helper(args, device)

    # def _init_classifier_training_helper(self, args, device) -> None:
        # self.loss_contribution_mode = args.classifier_loss
        # self.early_exit_warmup = args.early_exit_warmup
        # self.classifier_training_helper = ClassifierTrainingHelper(self.net, self.loss_contribution_mode, args.G, device)

    # def _init_gate_training_helper(self, args, device) -> None:

        # self.gate_training_helper = GateTrainingHelper(self.net, args.gate_objective, args.G, device)

    # def get_surrogate_loss(self, inputs, targets, training_phase=None):
        # if self.net.training:
        #     self.optimizer.zero_grad()
        #     if training_phase == TrainingPhase.CLASSIFIER:
        #         return self.classifier_training_helper.get_loss(inputs, targets)
        #     elif training_phase == TrainingPhase.GATE:
        #         return self.gate_training_helper.get_loss(inputs, targets)
        # else:
        #     with torch.no_grad():
        #         classifier_loss, things_of_interest = self.classifier_training_helper.get_loss(inputs, targets)
        #         gate_loss, things_of_interest_gate = self.gate_training_helper.get_loss(inputs, targets)
        #         loss = (gate_loss + classifier_loss) / 2
        #         things_of_interest.update(things_of_interest_gate)
        #         return loss, things_of_interest


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
                mlflow.log_metrics(log_dict,
                                   step=batch_idx +
                                        (e * len(train_loader)))
            self.scheduler.step()

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

