from utils import DGraphFin, GraphSAGE


import torch
import torch_geometric.transforms as T


# 1.Device and save path arguments
device = 0
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
path='./datasets/632d74d4e2843a53167ee9a1-momodel/'

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

# 3.Initialize model
model_para = {
    'in_channels': data.x.size(-1),
    'hidden_channels': 64,
    'num_layers': 2,
    'out_channels': nlabels,
    # 'heads': 8,
    'dropout': 0.5,
    'batchnorm': False,
}
model_path = f'./results/GraphSAGE-layer{model_para["num_layers"]}-hid{model_para["hidden_channels"]}-drop{model_para["dropout"]}.pt'
out_path = f'./results/GraphSAGE-layer{model_para["num_layers"]}-hid{model_para["hidden_channels"]}-drop{model_para["dropout"]}-out.pt'
model = GraphSAGE(**model_para).to(device)
model.load_state_dict(
    torch.load(model_path, map_location=device)
)
out = model(data.x, data.adj_t)
torch.save(out, out_path)


def predict(data, node_id):
    """
    加载模型和模型预测
    :param node_id: int, 需要进行预测节点的下标
    :return: tensor, 类0以及类1的概率, torch.size[1,2]
    """

    # 模型预测时，测试数据已经进行了归一化处理
    # -------------------------- 实现模型预测部分的代码 ---------------------------
    out = torch.load(out_path, map_location=device)
    y_pred = out.exp()[node_id]  # (N,num_classes)

    return y_pred