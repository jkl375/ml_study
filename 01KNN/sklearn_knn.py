import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split

# 分类个数为 3
n_neighbors = 3

# 导入鸢尾花数据
iris = datasets.load_iris()
X = iris.data[:, :2] # 取前两行
y = iris.target # 分类值

# 划分训练测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


for weights in ['uniform', 'distance']:
	# 创建 knn 分类器
	clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
	clf.fit(x_train, y_train)
	
	
	Z = clf.predict(x_test)
	print('-------------------------------------')
	print(weights+'方法:')
	print('实际分类情况')
	print(y_test)
	print('knn分类情况')
	print(Z)

