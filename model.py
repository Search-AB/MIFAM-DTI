import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *
import csv


class MultiDeep(nn.Module):
    def __init__(self, nprotein, ndrug, nproteinfeat, ndrugfeat, nhid, nheads, alpha):
        """Dense version of GAT."""
        super(MultiDeep, self).__init__()

        self.protein_attentions1 = [GraphAttentionLayer(nproteinfeat, nhid, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.protein_attentions1):
            self.add_module('Attention_Protein1_{}'.format(i), attention)
        self.protein_MultiHead1 = [selfattention(nprotein, nhid, nprotein) for _ in range(nheads)]
        for i, attention in enumerate(self.protein_MultiHead1):
            self.add_module('Self_Attention_Protein1_{}'.format(i), attention)
        self.protein_prolayer1 = nn.Linear(nhid * nheads, nhid * nheads, bias=False)
        self.protein_LNlayer1 = nn.LayerNorm(nhid * nheads)

        self.protein_attentions2 = [GraphAttentionLayer(nhid * nheads, nhid, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.protein_attentions2):
            self.add_module('Attention_Protein2_{}'.format(i), attention)
        self.protein_MultiHead2 = [selfattention(nprotein, nhid, nprotein) for _ in range(nheads)]
        for i, attention in enumerate(self.protein_MultiHead2):
            self.add_module('Self_Attention_Protein2_{}'.format(i), attention)
        self.protein_prolayer2 = nn.Linear(nhid * nheads, nhid * nheads, bias=False)
        self.protein_LNlayer2 = nn.LayerNorm(nhid * nheads)

        self.drug_attentions1 = [GraphAttentionLayer(ndrugfeat, nhid, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.drug_attentions1):
            self.add_module('Attention_Drug1_{}'.format(i), attention)
        self.drug_MultiHead1 = [selfattention(ndrug, nhid, ndrug) for _ in range(nheads)]
        for i, attention in enumerate(self.drug_MultiHead1):
            self.add_module('Self_Attention_Drug1_{}'.format(i), attention)
        self.drug_prolayer1 = nn.Linear(nhid * nheads, nhid * nheads, bias=False)
        self.drug_LNlayer1 = nn.LayerNorm(nhid * nheads)

        self.drug_attentions2 = [GraphAttentionLayer(nhid * nheads, nhid, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.drug_attentions2):
            self.add_module('Attention_Drug2_{}'.format(i), attention)
        self.drug_MultiHead2 = [selfattention(ndrug, nhid, ndrug) for _ in range(nheads)]
        for i, attention in enumerate(self.drug_MultiHead2):
            self.add_module('Self_Attention_Drug2_{}'.format(i), attention)
        self.drug_prolayer2 = nn.Linear(nhid * nheads, nhid * nheads, bias=False)
        self.drug_LNlayer2 = nn.LayerNorm(nhid * nheads)

        self.FClayer1 = nn.Linear(nhid * nheads * 2, nhid * nheads * 2)
        self.FClayer2 = nn.Linear(nhid * nheads * 2, nhid * nheads * 2)
        self.FClayer3 = nn.Linear(nhid * nheads * 2, 1)
        self.output = nn.Sigmoid()

    def forward(self, protein_features, protein_adj, drug_features, drug_adj, idx_protein_drug, device):
        proteinx = torch.cat([att(protein_features, protein_adj) for att in self.protein_attentions1], dim=1)
        proteinx = self.protein_prolayer1(proteinx)
        proteinayer = proteinx
        temp = torch.zeros_like(proteinx)
        for selfatt in self.protein_MultiHead1:
            temp = temp + selfatt(proteinx)
        proteinx = temp + proteinayer
        proteinx = self.protein_LNlayer1(proteinx)

        proteinx = torch.cat([att(proteinx, protein_adj) for att in self.protein_attentions2], dim=1)
        proteinx = self.protein_prolayer2(proteinx)
        proteinayer = proteinx
        temp = torch.zeros_like(proteinx)
        for selfatt in self.protein_MultiHead2:
            temp = temp + selfatt(proteinx)
        proteinx = temp + proteinayer
        proteinx = self.protein_LNlayer2(proteinx)

        # proteinx = torch.cat([att(proteinx, protein_adj) for att in self.protein_attentions2], dim=1)
        # proteinx = self.protein_prolayer2(proteinx)
        # proteinayer = proteinx
        # temp = torch.zeros_like(proteinx)
        # for selfatt in self.protein_MultiHead2:
        #     temp = temp + selfatt(proteinx)
        # proteinx = temp + proteinayer
        # proteinx = self.protein_LNlayer2(proteinx)

        drugx = torch.cat([att(drug_features, drug_adj) for att in self.drug_attentions1], dim=1)
        drugx = self.drug_prolayer1(drugx)
        druglayer = drugx
        temp = torch.zeros_like(drugx)
        for selfatt in self.drug_MultiHead1:
            temp = temp + selfatt(drugx)
        drugx = temp + druglayer
        drugx = self.drug_LNlayer1(drugx)

        drugx = torch.cat([att(drugx, drug_adj) for att in self.drug_attentions2], dim=1)
        drugx = self.drug_prolayer2(drugx)
        druglayer = drugx
        temp = torch.zeros_like(drugx)
        for selfatt in self.drug_MultiHead2:
            temp = temp + selfatt(drugx)
        drugx = temp + druglayer
        drugx = self.drug_LNlayer2(drugx)

        # drugx = torch.cat([att(drugx, drug_adj) for att in self.drug_attentions2], dim=1)
        # drugx = self.drug_prolayer2(drugx)
        # druglayer = drugx
        # temp = torch.zeros_like(drugx)
        # for selfatt in self.drug_MultiHead2:
        #     temp = temp + selfatt(drugx)
        # drugx = temp + druglayer
        # drugx = self.drug_LNlayer2(drugx)

        protein_drug_x = torch.cat((proteinx[idx_protein_drug[:, 0]], drugx[idx_protein_drug[:, 1]]), dim=1)
        protein_drug_x = protein_drug_x.to(device)
        protein_drug_x = self.FClayer1(protein_drug_x)
        protein_drug_x = F.relu(protein_drug_x)
        protein_drug_x = self.FClayer2(protein_drug_x)
        protein_drug_x = F.relu(protein_drug_x)
        protein_drug_x = self.FClayer3(protein_drug_x)
        protein_drug_x = protein_drug_x.squeeze(-1)
        return protein_drug_x

