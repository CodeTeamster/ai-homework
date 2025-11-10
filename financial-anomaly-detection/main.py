from utils import DGraphFin, MLP
from utils.evaluator import Evaluator
from torch_geometric.nn.models import GraphSAGE

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T


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


# 3.Initialize model
model_para = {
    'in_channels': data.x.size(-1),
    'hidden_channels': 128,
    'num_layers': 2,
    'out_channels': nlabels,
    'dropout': 0.7,
    # 'batchnorm': False,
}
# model = MLP(**model_para).to(device)
model = GraphSAGE(**model_para).to(device)
model.load_state_dict(
    torch.load('./results/GraphSAGE-layer2-hid128-drop0.pt', map_location=device)
)
evaluator = Evaluator(eval_metric='auc')


def predict(data, node_id):
    """
    加载模型和模型预测
    :param node_id: int, 需要进行预测节点的下标
    :return: tensor, 类0以及类1的概率, torch.size[1,2]
    """

    # 模型预测时，测试数据已经进行了归一化处理
    # -------------------------- 实现模型预测部分的代码 ---------------------------
    with torch.no_grad():
        model.eval()
        x = data.x[node_id].unsqueeze(0) if len(data.x[node_id].shape) == 1 else data.x[node_id]
        out = F.log_softmax(
            model(
                x = x,
                edge_index = data.adj_t[node_id, node_id]
            ),
            dim=-1,
        )
        out = out.squeeze(0) if out.dim() == 2 and out.shape[0] == 1 else out
        # out = out.squeeze(0) if out.size(0) == 1 else out
        # [0] -> torch.Size([1, 2])
        # [0, 1] -> torch.Size([2, 2])
        y_pred = out.exp()  # (N,num_classes)

    return y_pred


# def predict(data, node_id):
#     """
#     加载模型和模型预测
#     :param node_id: int, 需要进行预测节点的下标
#     :return: tensor, 类0以及类1的概率, torch.size[1,2]
#     """

#     # 模型预测时，测试数据已经进行了归一化处理
#     # -------------------------- 实现模型预测部分的代码 ---------------------------
#     with torch.no_grad():
#         model.eval()
#         out = model(data.x[node_id])
#         # 0 -> torch.Size([2])
#         # [0] -> torch.Size([1, 2])
#         # [0, 1] -> torch.Size([2, 2])
#         y_pred = out.exp()  # (N,num_classes)

#     return y_pred


# node_idx = torch.cat((data.valid_mask[0:300:3], data.valid_mask[10000:10300:3]), dim=0)
node_idx = data.test_mask
y_pred = torch.argmax(predict(data, node_idx), dim=-1)
data.y[node_idx] = y_pred
y_true = data.y[node_idx]
# y_pred = y_pred.unsqueeze(0) if y_pred.dim() == 1 else y_pred
# y_true = torch.tensor([data.y[node_idx]], device=device) if data.y[node_idx].numel() == 1 else data.y[node_idx]
print(dataset[0].y[node_idx])
DGraphFin.save(dataset, './datasets/data.pt')
# print(
#     evaluator.eval(
#         y_true,
#         y_pred,
#     )[evaluator.eval_metric]
# )

# node_idx = 1
# y_pred = predict(data, node_idx)
# y_pred = y_pred.unsqueeze(0) if y_pred.dim() == 1 else y_pred
# y_true = torch.tensor([data.y[node_idx]], device=device)
# print(y_pred)
# print(y_true)
# print(
#     evaluator.eval(
#         y_true,
#         y_pred,
#     )[evaluator.eval_metric]
# )
# print(f'节点 {node_idx} 预测对应的标签为:{torch.argmax(y_pred)}')