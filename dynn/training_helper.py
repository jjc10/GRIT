import torch
from torch import nn
import numpy as np
#

criterion = nn.CrossEntropyLoss()

class LearningHelper:
    def __init__(self, net, optimizer) -> None:
        self.net = net
        self.optimizer = optimizer
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
        loss = criterion(final_logits[0], targets)  # the grad_fn of this loss should be None if frozen
        num_gates = len(intermediate_logits) + 1
        losses = []
        losses.append(loss)
        for l, intermediate_logit in enumerate(intermediate_logits):
            intermediate_loss = criterion(intermediate_logit[0], targets)
            losses.append(intermediate_loss)
            loss += (num_gates - l) * intermediate_loss # we scale the gradient by G-l => early gates have bigger gradient
        things_of_interest = {
            'intermediate_logits': intermediate_logits,
            'final_logits': final_logits,
            'losses': losses
        }
        return loss, things_of_interest

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

