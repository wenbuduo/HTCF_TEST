# -*- coding: utf-8 -*-
# Author : yuxiang Zeng

import time
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import *

import dgl as d
from dgl.nn.pytorch import SAGEConv
from utils.metamodel import MetaModel


class GraphCF(MetaModel):
    def __init__(self, user_num, serv_num, args):
        self.args = args
        super(GraphCF, self).__init__(user_num, serv_num, args)
        try:
            userg = pickle.load(open('./modules/models/baselines/userg.pk', 'rb'))
            servg = pickle.load(open('./modules/models/baselines/servg.pk', 'rb'))
        except:
            user_lookup, serv_lookup, userg, servg = create_graph()
            pickle.dump(userg, open('./modules/models/baselines/userg.pk', 'wb'))
            pickle.dump(servg, open('./modules/models/baselines/servg.pk', 'wb'))
        self.usergraph, self.servgraph = userg, servg
        self.dim = args.dimension
        self.order = args.order
        self.user_embeds = GraphSAGEConv(self.usergraph, args.dimension, args.order)
        self.item_embeds = GraphSAGEConv(self.servgraph, args.dimension, args.order)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2 * args.dimension, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, inputs, train):
        userIdx, itemIdx = inputs
        user_embeds = self.user_embeds(userIdx)
        serv_embeds = self.item_embeds(itemIdx)
        user_embeds = user_embeds.to(torch.float32)
        serv_embeds = serv_embeds.to(torch.float32)
        estimated = self.layers(torch.cat((user_embeds, serv_embeds), dim=-1)).sigmoid()
        estimated = estimated.reshape(user_embeds.shape[0])
        return estimated.flatten()

    def prepare_test_model(self):
        pass

    def get_embeds_parameters(self):
        parameters = []
        for params in self.user_embeds.parameters():
            parameters += [params]
        for params in self.item_embeds.parameters():
            parameters += [params]
        return parameters

    def get_attention_parameters(self):
        parameters = []
        for params in self.user_attention.parameters():
            parameters += [params]
        for params in self.item_attention.parameters():
            parameters += [params]
        return parameters

    def get_mlp_parameters(self):
        parameters = []
        for params in self.layers.parameters():
            parameters += [params]
        return parameters


class GraphSAGEConv(torch.nn.Module):
    def __init__(self, graph, dim, order=3):
        # 调用基类的构造函数
        super(GraphSAGEConv, self).__init__()
        # 设置GraphSAGE卷积的阶数
        self.order = order
        # 存储图，并使用Kaiming正态分布初始化节点嵌入
        self.graph = graph
        self.embedding = torch.nn.Parameter(torch.Tensor(self.graph.number_of_nodes(), dim))
        torch.nn.init.kaiming_normal_(self.embedding)
        # 将初始嵌入分配给图的节点特征，键为 'L0'
        self.graph.ndata['L0'] = self.embedding
        # 创建GraphSAGEConv层、LayerNorms和ELU激活函数的列表
        self.layers = torch.nn.ModuleList([SAGEConv(dim, dim, aggregator_type='gcn') for _ in range(order)])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(dim) for _ in range(order)])
        self.acts = torch.nn.ModuleList([torch.nn.ELU() for _ in range(order)])
    def forward(self, uid):
        # 如果可用，将图移到GPU
        g = self.graph.to('cuda')
        # 从 'L0' 获取初始节点特征
        feats = g.ndata['L0']
        # 对每一层执行GraphSAGE卷积
        for i, (layer, norm, act) in enumerate(zip(self.layers, self.norms, self.acts)):
            feats = layer(g, feats).squeeze()
            feats = norm(feats)
            feats = act(feats)
            g.ndata[f'L{i + 1}'] = feats
        # 提取指定用户（uid）的最终嵌入
        embeds = g.ndata[f'L{self.order}'][uid]
        return embeds


# GraphMF
def create_graph():
    # 创建空的用户图和服务图
    userg = d.graph([])
    servg = d.graph([])

    # 创建特征查找对象
    user_lookup = FeatureLookup()
    serv_lookup = FeatureLookup()

    # 读取用户列表数据和服务列表数据
    ufile = pd.read_csv('./datasets/原始数据/userlist_table.csv')
    sfile = pd.read_csv('./datasets/原始数据/wslist_table.csv')

    # 转换为NumPy数组
    ulines = ufile.to_numpy()
    slines = sfile.to_numpy()

    # 注册用户和服务的类别标识
    for i in range(339):
        user_lookup.register('User', i)
    for j in range(5825):
        serv_lookup.register('Serv', j)

    # 注册用户的额外特征类别
    for ure in ulines[:, 2]:#country
        user_lookup.register('URE', ure)
    for uas in ulines[:, 4]:#AS
        user_lookup.register('UAS', uas)

    # 注册服务的额外特征类别
    for sre in slines[:, 4]:#country
        serv_lookup.register('SRE', sre)
    for spr in slines[:, 2]:#service provider
        serv_lookup.register('SPR', spr)
    for sas in slines[:, 6]:#AS
        serv_lookup.register('SAS', sas)

    # 添加用户和服务节点
    userg.add_nodes(len(user_lookup))
    servg.add_nodes(len(serv_lookup))

    # 添加用户之间的边
    for line in ulines:
        uid = line[0]
        ure = user_lookup.query_id(line[2])
        if not userg.has_edges_between(uid, ure):
            userg.add_edges(uid, ure)

        uas = user_lookup.query_id(line[4])
        if not userg.has_edges_between(uid, uas):
            userg.add_edges(uid, uas)

    # 添加服务之间的边
    for line in slines:
        sid = line[0]
        sre = serv_lookup.query_id(line[4])
        if not servg.has_edges_between(sid, sre):
            servg.add_edges(sid, sre)

        sas = serv_lookup.query_id(line[6])
        if not servg.has_edges_between(sid, sas):
            servg.add_edges(sid, sas)

        spr = serv_lookup.query_id(line[2])
        if not servg.has_edges_between(sid, spr):
            servg.add_edges(sid, spr)

    # 对图进行自循环和双向化处理
    userg = d.add_self_loop(userg)
    userg = d.to_bidirected(userg)
    servg = d.add_self_loop(servg)
    servg = d.to_bidirected(servg)

    return user_lookup, serv_lookup, userg, servg


class FeatureLookup:
    def __init__(self):
        # 初始化计数器、内部存储和相关映射
        self.__inner_id_counter = 0
        self.__inner_bag = {}
        self.__category = set()
        self.__category_bags = {}
        self.__inverse_map = {}

    def register(self, category, value):
        # 将值注册到特定类别中
        self.__category.add(category)
        # 如果类别不存在，则创建一个新的类别子树
        if category not in self.__category_bags:
            self.__category_bags[category] = {}
        # 如果值在全局索引中不存在，则创建之，同时更新映射和类别子树
        if value not in self.__inner_bag:
            self.__inner_bag[value] = self.__inner_id_counter
            self.__inverse_map[self.__inner_id_counter] = value
            if value not in self.__category_bags[category]:
                self.__category_bags[category][value] = self.__inner_id_counter
            self.__inner_id_counter += 1

    def query_id(self, value):
        # 查询值的索引
        return self.__inner_bag[value]

    def query_value(self, id):
        # 查询索引对应的值
        return self.__inverse_map[id]

    def __len__(self):
        # 返回内部存储的长度
        return len(self.__inner_bag)



