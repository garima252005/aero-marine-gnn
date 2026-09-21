import torch, torch_geometric
print(torch.__version__, torch_geometric.__version__)
from torch_geometric.data import HeteroData
print("HeteroData OK")