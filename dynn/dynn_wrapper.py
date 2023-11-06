import torch
import torch.nn as nn
from grit.head.san_graph import SANGraphHead
from torch_geometric.graphgym.models import GNNGraphHead
import importlib
class DynnWrapper(nn.Module):

    def __init__(self, grit_transformer, head_dim_in, head_dim_out):
        super().__init__()
        self.grit_transformer = grit_transformer
        self.head_dim_in = head_dim_in
        self.head_dim_out = head_dim_out

    def set_intermediate_heads(self, intermediate_head_positions):
        self.intermediate_head_positions = intermediate_head_positions
        self.num_of_gates = len(intermediate_head_positions)
        head = self.grit_transformer.post_mp
        assert self._is_supported_graph_head(type(head).__name__), "Unsupported head type, make sure you add support for it"
        self.intermediate_heads = nn.ModuleList([
            SANGraphHead(dim_in = head.dim_in,
                         dim_out=head.dim_out,
                         L=head.L) if type(head).__name__ == 'SANGraphHead' else GNNGraphHead(self.head_dim_in, self.head_dim_out)
            for _ in range(len(self.intermediate_head_positions))])


    def forward(self, batch):
        # As regular forward but pass through IMs
        inter_outs = []
        for name, module in self.grit_transformer.named_children():
            if name == 'layers':
                for layer_idx in range(len(module)):
                    intermediate_layer = module[layer_idx]
                    batch = intermediate_layer(batch)
                    if layer_idx < len(self.intermediate_heads):
                        intermediate_head = self.intermediate_heads[layer_idx]
                        inter_out = intermediate_head(batch)[0]
                        inter_outs.append(inter_out)
            else:
                batch = module(batch)
        return batch[0], inter_outs

    def _is_supported_graph_head(self, head_class_name):
        return head_class_name == 'SANGraphHead' or head_class_name == 'GNNGraphHead'

    def cost_per_exit(self):
        number_of_layers = len(self.intermediate_heads) + 1
        return [i / number_of_layers for i in range(number_of_layers)]
# Verify model is properly frozen list(filter(lambda x: x[1].requires_grad, list(self.named_parameters())))