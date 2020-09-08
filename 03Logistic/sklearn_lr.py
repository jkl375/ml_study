import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split


# 导入鸢尾花数据
iris = datasets.load_iris()
X = iris.data[:, :3] # 取前3行
y = iris.target # 分类值

# 划分训练测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

