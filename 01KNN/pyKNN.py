import numpy as np
import operator

class KNN():
	def __init__(self, X_text, dataSet, labels, k):
		self.X_text = X_text
		self.dataSet = dataSet
		self.labels = labels
		self.k = k
	
	# -----------实现 KNN 方法的第一种方式-----------------
	
	def knn(self):
		'''
		1.计算距离
		2.k个最近的标签
		3.出现次数最多的标签即为最终类别
		:return:
		'''
		x_size = self.X_text.shape[0]
		sortedCount = []
		# 对测试集分行
		for i in range(x_size):
			dataSetSize = self.dataSet.shape[0]
			# 计算距离
			distances = (((np.tile(self.X_text[i,:], (dataSetSize, 1)) - self.dataSet) ** 2).sum(axis=1)) ** 0.5
			sortedDistIndicies = distances.argsort()
			# 选择距离最小的k个点
			classCount = {}
			for j in range(self.k):
				voteIlabel = self.labels[sortedDistIndicies[j]]
				# print(voteIlabel)
				classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
			# 排序并返回出现最多的那个类型
			sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
			sortedCount.append(sortedClassCount[0][0])
		return sortedCount
		
		
		
	
	
	
	
if __name__ == "__main__":
	dataSet = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	X_text = np.array([[0.1, 0.1], [1.0, 1.0]])
	k = 3
	
	kn = KNN(X_text, dataSet, labels, k)
	print(kn.knn())
	
	