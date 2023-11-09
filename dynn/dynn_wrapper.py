import torch
import torch.nn as nn
from grit.head.san_graph import SANGraphHead
from torch_geometric.graphgym.models import GNNGraphHead
from gates.gate import Gate, GateType
from gates.identity_gate import IdentityGate
from gates.learnable_uncertainty_gate import LearnableUncGate
import importlib
class DynnWrapper(nn.Module):

    def __init__(self, grit_transformer, head_dim_in, head_dim_out, ce_ic_tradeoff):
        super().__init__()
        self.grit_transformer = grit_transformer
        self.head_dim_in = head_dim_in
        self.head_dim_out = head_dim_out
        self.CE_IC_tradeoff = ce_ic_tradeoff

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

    def get_gate_prediction(self, l, current_logits):
        return self.gates[l](current_logits)

    def set_learnable_gates(self, gate_positions, direct_exit_prob_param=False, gate_type=GateType.UNCERTAINTY):
        self.gate_positions = gate_positions
        self.direct_exit_prob_param = direct_exit_prob_param
        self.gate_type = gate_type
        if gate_type == GateType.UNCERTAINTY:
            self.gates = nn.ModuleList([
                LearnableUncGate() for _ in range(len(self.gate_positions))])
        elif gate_type == GateType.IDENTITY:
            self.gates = nn.ModuleList([IdentityGate() for _ in range(len(self.gate_positions))])
    def cost_per_exit(self):
        number_of_layers = len(self.intermediate_heads) + 1
        return [i / number_of_layers for i in range(1, number_of_layers + 1)]
# Verify model is properly frozen list(filter(lambda x: x[1].requires_grad, list(self.named_parameters())))