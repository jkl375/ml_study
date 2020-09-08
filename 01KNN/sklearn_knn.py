import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets

n_neightbors = 3

# 导入鸢尾花数据
iris = datasets.load_iris()
X = iris.data[:, :2] # 取前两行
y = iris.target # 分类值

# 步长
h = 0.01


