###############################################################################
# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder
# in your jobs
###############################################################################
from utils import DGraphFin, MLP
from utils.evaluator import Evaluator
from torch_geometric.nn.models import GAT, GraphSAGE

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T


def train(model, data, train_idx, optimizer):
     # data.y is labels of shape (N, )
    model.train()
    optimizer.zero_grad()
    out = F.log_softmax(
        model(
            x=data.x[train_idx],
            edge_index=data.adj_t[train_idx, train_idx]
        ),
        dim=-1,
    )
    loss = F.nll_loss(out, data.y[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model, data, split_idx, evaluator):
    # data.y is labels of shape (N, )
    with torch.no_grad():
        model.eval()

        losses, eval_results = dict(), dict()
        for key in ['train', 'valid']:
            node_id = split_idx[key]

            out = F.log_softmax(
                model(
                    x = data.x[node_id],
                    edge_index = data.adj_t[node_id, node_id]
                ),
                dim=-1,
            )
            y_pred = out.exp()  # (N,num_classes)

            losses[key] = F.nll_loss(out, data.y[node_id]).item()
            eval_results[key] = evaluator.eval(
                data.y[node_id],
                y_pred,
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
    transform=T.ToSparseTensor()
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
# The test labels were assigned based on the model's predictions.
data.y[data.test_mask] = 0
train_idx = torch.cat((split_idx['train'], split_idx['test']), dim=0)


# 3.Initialize model, optimizer and evaluator
model_para = {
    'in_channels': data.x.size(-1),
    'hidden_channels': 64,
    'num_layers': 3,
    'out_channels': nlabels,
    # 'heads': 8,
    'dropout': 0.5,
    # 'v2': False,
    # 'batchnorm': False,
}
optim_para = {
    'lr': 0.01,
    'weight_decay': 5e-6,
}
# model = MLP(**model_para).to(device)
model = GraphSAGE(**model_para).to(device)
model.reset_parameters()
evaluator = Evaluator(eval_metric='auc')
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=optim_para['lr'],
    weight_decay=optim_para['weight_decay']
)


# 4.Training
epochs = 3000
log_steps = 10
max_valid_eval = 0
min_valid_loss = 1e8
for epoch in range(1,epochs + 1):
    loss = train(model, data, train_idx, optimizer)
    eval_results, losses, out = test(model, data, split_idx, evaluator)
    train_eval, valid_eval = eval_results['train'], eval_results['valid']
    train_loss, valid_loss = losses['train'], losses['valid']

    if valid_eval > max_valid_eval:
        max_valid_eval = valid_eval
        torch.save(model.state_dict(), save_dir+'/GraphSAGE-layer3-hid64-drop0.5-test.pt')

    if epoch % log_steps == 0:
        print(f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_eval:.3f}, '
              f'Valid: {100 * valid_eval:.3f} ')