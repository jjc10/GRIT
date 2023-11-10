'''Train BOOSTED baseline from checkpoint of trained backbone'''

import torch
import mlflow

criterion = torch.nn.CrossEntropyLoss()
DISPLAY_COUNT = 100
def train_boosted(net, device, train_loader, optimizer, epoch):
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch
        targets = batch.y
        inputs, targets = inputs.to(device), targets.to(device)
        boosted_loss = get_boosted_loss(inputs, targets, optimizer, net)
        boosted_loss.backward()
        optimizer.step()
        if batch_idx % DISPLAY_COUNT == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}. Loss {boosted_loss}')


def test_boosted(net, test_loader, device):
    net.eval()
    n_blocks = net.num_layers
    corrects = [0] * n_blocks
    totals = [0] * n_blocks
    for batch_idx, batch in enumerate(test_loader):
        x = batch
        y = batch.y
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            preds = net.forward(x)
   
        for i, pred in enumerate(preds):
            corrects[i] += (torch.argmax(pred, 1) == y).sum().item()
            totals[i] += y.shape[0]
    corrects = [c / t * 100 for c, t in zip(corrects, totals)]
    log_dict = {}
    for blk in range(n_blocks):
        log_dict['test' + '/acc' +
                 str(blk)] = corrects[blk]
    mlflow.log_metrics(log_dict)
    return corrects

def get_boosted_loss(inputs, targets, optimizer, boosted_wrapper):
    # assert isinstance(net, Boosted_T2T_ViT), 'Boosted loss only available for boosted t2t vit'
    n_blocks = len(boosted_wrapper.intermediate_head_positions)
    optimizer.zero_grad()

    # Ensembling
    preds, pred_ensembles = boosted_wrapper.forward_all(inputs, n_blocks - 1)
    loss_all = 0
    size = len(targets)
    for stage in range(n_blocks):
        # train weak learner
        # fix F
        with torch.no_grad():
            if not isinstance(pred_ensembles[stage], torch.Tensor):

                out = torch.unsqueeze(torch.Tensor([pred_ensembles[stage]]), 0)  # 1x1
                out = out.expand(size, boosted_wrapper.num_classes).cuda()
            else:
                out = pred_ensembles[stage]
            out = out.detach()
        loss = criterion(preds[stage] + out, targets)
        loss_all = loss_all + loss
    return loss_all
