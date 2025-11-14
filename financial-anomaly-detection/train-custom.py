from torch_geometric.nn import SAGEConv
from utils import DGraphFin
from utils.evaluator import Evaluator


import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=3,
        dropout=0.5,
        batchnorm=False
    ):
        super(GraphSAGE, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.batchnorm = batchnorm

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if batchnorm else None

        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            if batchnorm:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
                if batchnorm:
                    self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)
    loss = F.cross_entropy(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, data, split_idx, evaluator):
    with torch.no_grad():
        model.eval()
        losses, eval_results = dict(), dict()
        for key in ['train', 'valid']:
            node_id = split_idx[key]
            out = model(data.x, data.adj_t)
            y_pred = out.exp()  # (N, num_classes)
            losses[key] = F.cross_entropy(out[node_id], data.y[node_id]).item()
            eval_results[key] = evaluator.eval(
                data.y[node_id],
                y_pred[node_id]
            )[evaluator.eval_metric]

    return eval_results, losses, y_pred


# 1.Device and save path arguments
device = 0
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
path='./datasets/632d74d4e2843a53167ee9a1-momodel/'
save_dir='./results/'

# 2.Datasets
dataset_name='DGraph'
dataset = DGraphFin(
    root=path,
    name=dataset_name,
    transform=T.ToSparseTensor(remove_edge_index=False)
).to(device)
nlabels = dataset.num_classes
if dataset_name in ['DGraph']:
    nlabels = 2
data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()
if dataset_name in ['DGraph']:
    x = data.x
    data.x = (x - x.mean(0)) / x.std(0)
if data.y.dim() == 2:
    data.y = data.y.squeeze(1)
split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}
train_idx = split_idx['train']

# 3.Initialize model, optimizer and evaluator
model_para = {
    'in_channels': data.x.size(-1),
    'hidden_channels': 64,
    'num_layers': 2,
    'out_channels': nlabels,
    # 'heads': 8,
    'dropout': 0.5,
    # 'v2': False,
    'batchnorm': False,
}
optim_para = {
    'lr': 1e-2,
    'weight_decay': 5e-4,
}
model = GraphSAGE(**model_para).to(device)
evaluator = Evaluator(eval_metric='auc')
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=optim_para['lr'],
    weight_decay=optim_para['weight_decay']
)

# 4.Training
epochs = 2000
log_steps = 1
best_valid = 0
min_valid_loss = 1e8
for epoch in range(1, epochs + 1):
    loss = train(model, data, train_idx, optimizer)
    eval_results, losses, out = test(model, data, split_idx, evaluator)

    train_eval, valid_eval = eval_results['train'], eval_results['valid']
    train_loss, valid_loss = losses['train'], losses['valid']
    if valid_loss < min_valid_loss:
        min_valid_loss = valid_loss
        torch.save(model.state_dict(), os.path.join(save_dir, 'GraphSAGE-layer2-hid64-drop0.5.pt'))

    if epoch % log_steps == 0:
        print(f'Epoch: {epoch:02d}, '
                f'Loss: {loss:.4f}, '
                f'Train: {100 * train_eval:.3f}, '
                f'Valid: {100 * valid_eval:.3f} ')