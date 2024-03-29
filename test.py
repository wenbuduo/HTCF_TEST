import dgl

# 定义边缘列表
src = [0, 1, 2, 3]  # 源节点列表
dst = [2, 3, 0, 1]  # 目标节点列表

# 创建图
g = dgl.graph((src, dst))
src, dst = g.edges
print(scr)