def hyper_gnn(self):
    uHyper, iHyper = self.uHyper, self.iHyper
    # 两层图卷积
    uEmbed_gcn, iEmbed_gcn = self.GCN(self.uEmbed_ini, self.iEmbed_ini, self.adj, self.tpadj)
    uEmbed0 = self.uEmbed_ini + uEmbed_gcn
    iEmbed0 = self.iEmbed_ini + iEmbed_gcn

    # 超图的构建过程
    uKey = self.prepareKey(uEmbed0)
    iKey = self.prepareKey(iEmbed0)

    ulats = [uEmbed0]
    ilats = [iEmbed0]

    for i in range(args.gnn_layer):
        self.propagate(ulats, uKey, uHyper)
        self.propagate(ilats, iKey, iHyper)

    ulat = torch.sum(torch.stack(ulats), dim=0)
    ilat = torch.sum(torch.stack(ilats), dim=0)

    return ulat, ilat


# 多跳节点之间的的信息传播
def GCN(self, ulat, ilat, adj, tpadj):
    ulats = [ulat]
    ilats = [ilat]

    adj = torch.as_tensor(adj)
    tpadj = torch.as_tensor(tpadj)

    for i in range(args.gcn_hops):
        temulat = torch.mm(adj, ilats[-1])
        temilat = torch.mm(tpadj, ulats[-1])
        ulats.append(temulat)
        ilats.append(temilat)
        # Stack and sum the list of tensors
    uEmbed_gcn = torch.sum(torch.stack(ulats[1:], dim=0), dim=0)
    iEmbed_gcn = torch.sum(torch.stack(ilats[1:], dim=0), dim=0)

    return uEmbed_gcn, iEmbed_gcn