import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 导入鸢尾花数据
iris = datasets.load_iris()
X = iris.data[:, :3] # 取前3行
y = iris.target # 分类值

# 划分训练测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# SVM分类器
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)

# 超平面
w = clf.coef_[0]
# 斜率
a = -w[0]/w[1]

y = clf.predict(x_test)
print('实际分类情况')
print(y_test)
print('svm分类情况')
print(y)
