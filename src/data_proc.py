import random

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pickle as pkl
from Bio import pairwise2
from sklearn.metrics.pairwise import cosine_similarity


# 将SMILES字符串转换为图数据
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILE string")

    # 提取原子特征
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(),  # 原子序数
            atom.GetDegree(),  # 成键数
            atom.GetTotalNumHs(),  # 附着氢原子数量
            atom.GetFormalCharge(),  # 价态
            int(atom.GetIsAromatic())  # 是否为芳环
        ])

    atom_features = torch.tensor(atom_features, dtype=torch.float)

    # 构建邻接矩阵
    edges = []
    for bond in mol.GetBonds():
        edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        edges.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))  # 无向图

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # 转换为PyG数据格式
    x = atom_features
    data = Data(x=x, edge_index=edge_index)
    return data


# 定义GCN层
class GCNLayer(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = torch.nn.Linear(in_feats * 2, out_feats)

    def forward(self, x, edge_index):
        # 聚合邻居节点特征
        agg = torch.zeros_like(x)
        if edge_index.numel() == 0:
            print("ZERO")
            return torch.zeros([20, 128]).to(device)
        agg[edge_index[0]] = x[edge_index[1]]
        out = self.linear(torch.cat((x, agg), dim=-1))
        return out


# 特征提取
def extract_features(smiles):
    graph_data = smiles_to_graph(smiles)
    gcn_layer = GCNLayer(in_feats=5, out_feats=128).to(device)  # 5个原子特征到128维嵌入
    node_embeddings = gcn_layer(graph_data.x.to(device), graph_data.edge_index.to(device))
    # 使用平均池化作为读出函数
    molecule_embedding = node_embeddings.mean(dim=0)
    return molecule_embedding


# 氨基酸编码
aa_index = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
    'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
    'X': 20, 'U': 21
}


# 将氨基酸序列转换为图数据
def aa_sequence_to_graph(sequence):
    # 提取氨基酸特征
    node_features = torch.tensor([aa_index[aa] for aa in sequence], dtype=torch.float).unsqueeze(1)

    # 构建邻接矩阵
    edges = []
    for i in range(len(sequence) - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))  # 无向图

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # 转换为PyG数据格式
    x = node_features
    data = Data(x=x, edge_index=edge_index)
    return data


# 定义GCN模型
class ProteinGCN(torch.nn.Module):
    def __init__(self):
        super(ProteinGCN, self).__init__()
        self.conv1 = GCNConv(1, 64)
        self.conv2 = GCNConv(64, 128)  # 20为氨基酸种类数

    def forward(self, data):
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# 特征提取
def extract_aa_features(sequence):
    graph_data = aa_sequence_to_graph(sequence)
    model = ProteinGCN().to(device)
    node_embeddings = model(graph_data.to(device))
    # 使用平均池化作为读出函数
    protein_embedding = node_embeddings.mean(dim=0)
    return protein_embedding


def calculate_drug_similarity(smiles_list):
    # 将SMILES转换为RDKit分子对象
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    # 计算每个分子的指纹
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in molecules]

    # 计算相似性矩阵
    similarity_matrix = []
    for i in range(len(molecules)):
        row = []
        for j in range(len(molecules)):
            if i != j:
                # 计算Tanimoto相似性
                similarity = AllChem.DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                row.append(similarity)
            else:
                row.append(1.0)  # 对角线元素设置为1
        similarity_matrix.append(row)
        # print(i)

    return np.array(similarity_matrix)


def calculate_protein_similarity(sequence_list):
    similarity_matrix = []
    for i in range(len(sequence_list)):
        row = []
        for j in range(len(sequence_list)):
            if i != j:
                # 计算Smith-Waterman相似性
                alignments = pairwise2.align.globalxx(sequence_list[i], sequence_list[j])
                alignment = alignments[0]  # 取最佳匹配
                similarity = alignment[2]  # 相似性分数
                row.append(similarity)
            else:
                row.append(1.0)  # 对角线元素设置为1
        similarity_matrix.append(row)
        print(i)

    return np.array(similarity_matrix)


dataset = "DrugBank"
device = "cuda:1"

full = pd.read_csv(f"../data/{dataset}/full.csv")
smiles_list = []
protein_sequences = []

for i, row in full.iterrows():
    if row['SMILES'] not in smiles_list:
        smiles_list.append(row['SMILES'])
    if row['Protein'] not in protein_sequences:
        protein_sequences.append(row['Protein'])

prot2index = {}
drug2index = {}
for i, d in enumerate(smiles_list):
    if d not in drug2index:
        drug2index[d] = len(drug2index)

for i, p in enumerate(protein_sequences):
    if p not in prot2index:
        prot2index[p] = len(drug2index) + len(prot2index)

pkl.dump(drug2index, open(f"../data/{dataset}/drug2index.pkl", "wb"))
pkl.dump(prot2index, open(f"../data/{dataset}/prot2index.pkl", "wb"))

positive_pair_d = []
positive_pair_p = []
positive_pair = []
for i, row in full.iterrows():
    if int(row['Y']) == 1:
        positive_pair_d.append(drug2index[row['SMILES']])
        positive_pair_p.append(prot2index[row['Protein']])
        positive_pair.append((drug2index[row['SMILES']], prot2index[row['Protein']]))

negative_pair_d = []
negative_pair_p = []
ind_d = list(drug2index.values())
ind_p = list(prot2index.values())
for _ in range(len(positive_pair)):
    i_d = random.choice(ind_d)
    i_p = random.choice(ind_p)
    if (i_d, i_p) not in positive_pair:
        negative_pair_d.append(i_d)
        negative_pair_p.append(i_p)

Allnode = [i for i in range(len(drug2index) + len(prot2index))]
Allnode_df = pd.DataFrame({'node': Allnode})

print(f'pos pair: {len(positive_pair)}, neg pair: {len(negative_pair_d)}')

DrPrNum_Drpr = pd.DataFrame({'pair_d': positive_pair_d, 'pair_p': positive_pair_p})
DrPrNum_Drpr.to_csv(f"../data/{dataset}/DrPrNum_DrPr.csv", index=False, header=False)
AllNegative_DrPr = pd.DataFrame({'pair_d': negative_pair_d, 'pair_p': negative_pair_p})
AllNegative_DrPr.to_csv(f"../data/{dataset}/AllNegative_DrPr.csv", index=False, header=False)
Allnode_df.to_csv(f"../data/{dataset}/Allnode_DrPr.csv", index=False, header=False)

num = {"drug_num": len(drug2index), "prot_num": len(prot2index)}
pkl.dump(num, open(f"../data/{dataset}/num.pkl", "wb"))

drug_embeddings = []
for i, d in enumerate(smiles_list):
    drug_embeddings.append(extract_features(d).cpu().detach().numpy())

protein_embeddings = []
for i, p in enumerate(protein_sequences):
    protein_embeddings.append(extract_aa_features(p).cpu().detach().numpy())

prot_similarity_matrix = cosine_similarity(protein_embeddings)
drug_similarity_matrix = calculate_drug_similarity(smiles_list)
# prot_similarity_matrix = calculate_protein_similarity(protein_sequences)

pkl.dump(drug_similarity_matrix, open(f"../data/{dataset}/drug_similarity_matrix.pkl", "wb"))
pkl.dump(prot_similarity_matrix, open(f"../data/{dataset}/prot_similarity_matrix.pkl", "wb"))
AllNodeAttribute_DrPr = pd.concat([pd.DataFrame(drug_embeddings), pd.DataFrame(protein_embeddings)], axis=0)
AllNodeAttribute_DrPr.to_csv(f"../data/{dataset}/AllNodeAttribute_DrPr.csv", index=False, header=False)

prot_edge1 = []
prot_edge2 = []
drug_edge1 = []
drug_edge2 = []

for i, row in pd.read_csv(f'../data/{dataset}/DrPrNum_DrPr.csv', header=None).iterrows():
    prot_edge1.append(int(row[0]))
    prot_edge2.append(int(row[1]))
    drug_edge1.append(int(row[0]))
    drug_edge2.append(int(row[1]))

print(f"{dataset} positive edges: {len(prot_edge1)}")

pp = 0
p_e1 = []
p_e2 = []
for i in range(len(prot_similarity_matrix)):
    for j in range(i + 1, len(prot_similarity_matrix)):
        if prot_similarity_matrix[i][j] > 0.985:
            p_e1.append(i)
            p_e2.append(j)
            pp += 1

random_array = np.random.randint(0, len(p_e1) - 1, size=min(500, len(p_e1) - 1))
for ind in random_array:
    prot_edge1.append(p_e1[ind])
    prot_edge2.append(p_e2[ind])

print("Prot-Prot edges: ", pp)
df = pd.DataFrame({'0': prot_edge1, '1': prot_edge2})
df.to_csv(f'../data/{dataset}/prot_edge.csv', index=False, header=False)
print("P edges: ", len(prot_edge1))

dd = 0
d_e1 = []
d_e2 = []
for i in range(len(drug_similarity_matrix)):
    for j in range(i + 1, len(drug_similarity_matrix)):
        if drug_similarity_matrix[i][j] > 0.988:
            d_e1.append(i)
            d_e2.append(j)
            dd += 1

random_array = np.random.randint(0, len(d_e1) - 1, size=min(500, len(d_e1) - 1))
for ind in random_array:
    prot_edge1.append(d_e1[ind])
    prot_edge2.append(d_e2[ind])
    drug_edge1.append(d_e1[ind])
    drug_edge2.append(d_e2[ind])

print("Drug-Drug edges: ", dd)
print("Total edges: ", len(prot_edge1))
print("D edges: ", len(drug_edge1))

df = pd.DataFrame({'0': drug_edge1, '1': drug_edge2})
df.to_csv(f'../data/{dataset}/drug_edge.csv', index=False, header=False)

df = pd.DataFrame({'0': prot_edge1, '1': prot_edge2})
df.to_csv(f'../data/{dataset}/drug_prot_edge.csv', index=False, header=False)

print("Done")
