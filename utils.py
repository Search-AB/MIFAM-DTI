import numpy as np
import torch
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


def multiomics_data():

    #protein_fea
    DC = np.genfromtxt("./data/protein_fea/fea_DC.csv", delimiter=',', skip_header=1, dtype=np.dtype(str))
    PC = np.genfromtxt("./data/protein_fea/fea_phy_chem.csv", delimiter=',', skip_header=1, dtype=np.dtype(str))

    protein = DC[:, 0]

    DC = multiomics_fusion(DC, protein)
    PC = multiomics_fusion(PC, protein)
    DC = scale(DC)
    PC = scale(PC)

    pca = PCA(n_components=128)
    DC = pca.fit_transform(np.array(DC, dtype=float))
    pca = PCA(n_components=128)
    PC = pca.fit_transform(np.array(PC, dtype=float))

    #protein_adj
    ppi_adj = get_adj_array("./data/proteinprotein.txt")



    protein_number = len(protein)
    DC_adj = sim_graph(DC, protein_number)
    PC_adj = sim_graph(PC,protein_number)
    fusion_protein_adj2 = DC_adj+PC_adj
    fusion_protein_adj2 = fusion_protein_adj2 / np.max(fusion_protein_adj2)
    fusion_protein_adj = np.logical_or(DC_adj,PC_adj).astype(int)
    fusion_protein_fea = np.concatenate((DC,PC),axis=1)

    fingerprint = np.genfromtxt("./data/drug_fea/fea_FCFP.csv", delimiter=',', skip_header=1, dtype=np.dtype(str))
    physicochemical = np.genfromtxt("./data/drug_fea/fea_phy_chem.csv", delimiter=',',skip_header=1,dtype=np.dtype(str))
    drugdrug_adj = get_adj_array("./data/drugdrug.txt")
    drug = fingerprint[:,0]

    fingerprint = np.array(fingerprint)
    physicochemical = np.array(physicochemical)

    fingerprint = scale(np.array(fingerprint[:, 1:], dtype=float))
    pca = PCA(n_components=128)
    fingerprint = pca.fit_transform(fingerprint)

    physicochemical = scale(np.array(physicochemical[:, 1:], dtype=float))
    pca = PCA(n_components=128)
    physicochemical = pca.fit_transform(physicochemical)

    drug_number = len(drug)
    physicochemical_adj = sim_graph(physicochemical, drug_number)
    fingerprint_adj = sim_graph(fingerprint,drug_number)
    fusion_drug_adj2 = physicochemical_adj + fingerprint_adj
    fusion_drug_adj2 = fusion_drug_adj2 / np.max(fusion_drug_adj2)
    fusion_drug_adj = np.logical_or(physicochemical_adj,fingerprint_adj).astype(int)
    fusion_drug_fea = np.concatenate((physicochemical,fingerprint),axis=1)

    # 从文件中加载数据
    labeltxt = np.genfromtxt("./data/drugProtein.txt", dtype=np.dtype(str))

    # 获取正样本
    positive_sample_list = []
    for i in range(labeltxt.shape[0]):
        for j in range(labeltxt.shape[1]):
            if labeltxt[i, j] == '1':
                positive_sample_list.append([j, i, 1])

    # 获取所有值为0的索引
    zero_indices = np.argwhere(labeltxt == '0')

    # 随机选择4978个负样本
    negative_samples = 4978
    negative_indices = np.random.choice(zero_indices.shape[0], negative_samples, replace=False)
    negative_samples_list = [[index[1], index[0], 0] for index in zero_indices[negative_indices]]

    labellist = positive_sample_list + negative_samples_list
    labellist = torch.Tensor(labellist)
    print("drug protein lable:", labellist.shape)


    protein_feat, protein_adj = torch.FloatTensor(fusion_protein_fea), torch.FloatTensor(fusion_protein_adj)
    drug_feat, drug_adj = torch.FloatTensor(fusion_drug_fea), torch.FloatTensor(fusion_drug_adj)
    return protein_feat, protein_adj, drug_feat, drug_adj, labellist


def multiomics_fusion(omics_data, protein_fusion):
    finalomics = []
    protein_index = omics_data[:, 0].tolist()
    for protein in protein_fusion:
        index = protein_index.index(protein)
        finalomics.append(np.array(omics_data[index, 1:], dtype=float))
    return np.array(finalomics)


def sim_graph(omics_data, protein_number):
    sim_matrix = np.zeros((protein_number, protein_number), dtype=float)
    adj_matrix = np.zeros((protein_number, protein_number), dtype=float)

    for i in range(protein_number):
        for j in range(i + 1):
            sim_matrix[i, j] = np.dot(omics_data[i], omics_data[j]) / (
                        np.linalg.norm(omics_data[i]) * np.linalg.norm(omics_data[j]))
            sim_matrix[j, i] = sim_matrix[i, j]

    for i in range(protein_number):
        topindex = np.argsort(sim_matrix[i])[-10:]
        for j in topindex:
            adj_matrix[i, j] = 1
    return adj_matrix


def drugs_sim_fingerprints(drug_feature, drug_number):
    sim_matrix = np.ones((drug_number, drug_number), dtype=float)
    adj_matrix = np.zeros((drug_number, drug_number), dtype=float)

    for i in range(drug_number):
        for j in range(i):
            sim_matrix[i, j] = Tanimoto(drug_feature[i], drug_feature[j])
            sim_matrix[j, i] = sim_matrix[i, j]

    for i in range(drug_number):
        topindex = np.argsort(sim_matrix[i])[-10:]
        for j in topindex:
            adj_matrix[i, j] = 1
    return adj_matrix


def Tanimoto(ifea, jfea):
    inter, union = 0, 0
    for i in range(ifea.shape[0]):
        if ifea[i] == 1 and jfea[i] == 1:
            inter += 1
    return inter / (np.sum(ifea) + np.sum(jfea) - inter)

def get_adj_array(file_path):
    # 读取txt文件
    file_path = file_path  # 替换为你的文件路径
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 提取邻接矩阵信息
    adj_matrix = []
    for line in lines:
        row = [float(x) for x in line.strip().split()]  # 假设邻接矩阵中的元素以空格分隔
        adj_matrix.append(row)

    # 将邻接矩阵转换为NumPy数组
    adj_array = np.array(adj_matrix)
    return adj_array