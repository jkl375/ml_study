import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 3

# 导入鸢尾花数据
iris = datasets.load_iris()
X = iris.data[:, :2] # 取前两行
y = iris.target # 分类值

# 步长
h = 0.02


# 创建彩色的图
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
	# 创建 knn 分类器
	clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
	clf.fit(X, y)
	
	x_min, x_max = min(X[:, 0]) - 1, max(X[:, 0]) + 1
	y_min, y_max = min(X[:, 1]) - 1, max(X[:, 1]) + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
	                     np.arange(y_min, y_max, h))
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	
	# 彩图
	Z = Z.reshape(xx.shape)
	plt.figure()
	plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
	
	# 绘制训练点
	
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.title("3-Class classification (k = %i, weights = '%s')"
	          % (n_neighbors, weights))

plt.show()