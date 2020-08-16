from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
from matplotlib import style 
import numpy as np
import pandas as pd
style.use("ggplot")

df = pd.read_excel("titanic.xls")
df.drop(["body","name"], 1, inplace=True)
df.fillna(0, inplace=True)
df._convert(numeric = True)
column =  (df.head().columns.values)


for col in column:
	if df[col].dtype != np.int64 and df[col].dtype != np.float64:
		column_contents = np.array(df[col].values.tolist())
		label_encoder = LabelEncoder()
		integer_encoded = label_encoder.fit_transform(column_contents)
		integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
		onehot_encoder = OneHotEncoder(sparse=False, categories = "auto")
		onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
		df[col] = integer_encoded
		
		
X = np.array(df.drop(["survived"], 1).astype(float))
y = np.array(df["survived"])
'''
clf = KMeans(n_cluster = 2)
clf.fit(X)
'''

#from scratch ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
v = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11],
              [1,3],
              [8,9],
              [0,3],
              [5,4],
              [6,4],])


class K_Means:
	def __init__(self, k=2, tol=0.002, max_iter=300):
		self.k = k
		self.tol = tol
		self.max_iter = max_iter

	def fit(self, data):
		self.centroids = {}

		for i in range(self.k):
			self.centroids[i] = data[i]

		for i in range(self.max_iter):
			self.classifications = {}

			for i in range(self.k):
				self.classifications[i] = []

			for featureset in data:
				distances = [np.linalg.norm(featureset-self.centroids[centr]) for centr in self.centroids]
				classification = distances.index(min(distances))
				self.classifications[classification].append(featureset)

			prev_centroids = dict(self.centroids)

			for classification in self.classifications:
				self.centroids[classification] = np.average(self.classifications[classification],axis=0)

			optimized = True

			for b in self.centroids:
				original_centroid = prev_centroids[b]
				current_centroid = self.centroids[b]
				if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
					optimized = False

			if optimized:
				break

	def predict(self,item):
		distances = [np.linalg.norm(item-self.centroids[i]) for i in self.centroids]
		classification = distances.index(min(distances))
		return classification

myclf = K_Means()
myclf.fit(v)

for centroid in myclf.centroids:
	plt.scatter(myclf.centroids[centroid][0], myclf.centroids[centroid][1],marker="o", color="b")

for classification in myclf.classifications:
	for featureset in myclf.classifications[classification]:
		plt.scatter(featureset[0],featureset[1], marker = "x", color = "r")

plt.show()

for item in v:
	print (myclf.predict(item))


	

































