import torch
import torch.nn as nn
from grit.head.san_graph import SANGraphHead
from torch_geometric.graphgym.models import GNNGraphHead
from gradient_rescale import GradientRescaleFunction
import numpy as np

class BoostedWrapper(nn.Module):

    def __init__(self, grit_transformer, head_dim_in, head_dim_out, num_classes = 10, ensemble_reweight = [0.5]):
        super().__init__()
        self.grit_transformer = grit_transformer
        self.head_dim_in = head_dim_in
        self.head_dim_out = head_dim_out
        self.ensemble_reweight = ensemble_reweight
        self.num_classes = num_classes

    def set_intermediate_heads(self, intermediate_head_positions):
        self.intermediate_head_positions = intermediate_head_positions
        self.num_of_gates = len(intermediate_head_positions)
        self.num_layers = len(intermediate_head_positions) + 1
        head = self.grit_transformer.post_mp
        assert self._is_supported_graph_head(type(head).__name__), "Unsupported head type, make sure you add support for it"
        self.intermediate_heads = nn.ModuleList([
            SANGraphHead(dim_in = head.dim_in,
                         dim_out=head.dim_out,
                         L=head.L) if type(head).__name__ == 'SANGraphHead' else GNNGraphHead(self.head_dim_in, self.head_dim_out)
            for _ in range(len(self.intermediate_head_positions))])
        assert len(self.ensemble_reweight) in [1, 2, self.num_layers]
        if len(self.ensemble_reweight) == 1:
            self.ensemble_reweight = self.ensemble_reweight * self.num_layers
        elif len(self.ensemble_reweight) == 2:
            self.ensemble_reweight = list(np.linspace(self.ensemble_reweight[0], self.ensemble_reweight[1], self.num_layers))


    def boosted_forward(self, batch):
        res = []
        for name, module in self.grit_transformer.named_children():
            if name == 'layers':
                for layer_idx in range(len(module)):
                    intermediate_layer = module[layer_idx]
                    batch = intermediate_layer(batch)
                    # TODO fix the out of bound issue
                    last_idx = len(batch) - 1
                    batch[last_idx] = gradient_rescale(batch[last_idx], 1.0 / (self.num_layers - layer_idx))
                    if layer_idx < len(self.intermediate_heads):
                        intermediate_head = self.intermediate_heads[layer_idx]
                        inter_out = intermediate_head(batch)[0] # take prediction
                        res.append(inter_out)
                    batch[last_idx] = gradient_rescale(batch[last_idx], (self.num_layers - layer_idx - 1))
            else:
                batch = module(batch)
        res.append(batch[0])
        return res
    def forward(self, x):
        outs = self.boosted_forward(x)
        preds = [0]
        for i in range(len(outs)):
            pred = outs[i] + preds[-1] * self.ensemble_reweight[i]
            preds.append(pred)
        preds = preds[1:]
        return preds

    def forward_all(self, x, stage=None): # from forward_all in dynamic net which itself calls forward (boosted_forward in our case)
        """Forward the model until block `stage` and get a list of ensemble predictions
        """
        nBlocks = len(self.intermediate_heads)
        assert 0 <= stage < nBlocks
        outs = self.boosted_forward(x)
        preds = [0]
        for i in range(len(outs)):
            pred = (outs[i] + preds[-1]) * self.ensemble_reweight[i]
            preds.append(pred)
            if i == stage:
                break
        return outs, preds
    def _is_supported_graph_head(self, head_class_name):
        return head_class_name == 'SANGraphHead' or head_class_name == 'GNNGraphHead'

    def cost_per_exit(self):
        number_of_layers = len(self.intermediate_heads) + 1
        return [i / number_of_layers for i in range(1, number_of_layers + 1)]

gradient_rescale = GradientRescaleFunction.apply