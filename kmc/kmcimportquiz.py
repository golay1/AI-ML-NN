import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

df = pd.read_csv('HandWrittenLetters1.csv')
df1 = pd.read_csv('target.csv')
data1 = df
target1=df1
target=target1[target1.columns[0]]
#print("data:(points, features):")
#print(data.shape)

# Project the data: this step will take several seconds
tsne = TSNE(n_components=2, init='random')
data = tsne.fit_transform(data1)



c=26 #clusters
#do kmeans
kmeans = KMeans(n_clusters=c)
clusters = kmeans.fit_predict(data)
print("kmeansclust:(clusters, features):")
print(kmeans.cluster_centers_.shape)

#cluster centers
'''fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)'''

#match with labels
from scipy.stats import mode
labels = np.zeros_like(clusters)
for i in range(c):
    mask = (clusters == i)
    labels[mask] = mode(target[mask])[0]

#accuracy
from sklearn.metrics import accuracy_score
print('accuracy:')
print(accuracy_score(target, labels))
np.savetxt("prediction.csv", labels, delimiter=",")

#confusion matrix
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(target, labels)
plt.figure(figsize = (12,10)) #resize
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');


