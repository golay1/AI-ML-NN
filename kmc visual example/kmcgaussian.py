## Initialisation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mean1 = [3,5]
mean2 = [-5,2]
mean3 = [1,-4]

std = 4
cov = [[2*std,0],[0,2*std]] # diagonal covariance, points lie on x or y-axis
#make cov symmetrical, with 2*std as value

#mean,covarriance,size
x1,y1 = np.random.multivariate_normal(mean1,cov,100).T
x2,y2 = np.random.multivariate_normal(mean2,cov,100).T
x3,y3 = np.random.multivariate_normal(mean3,cov,100).T

x=np.concatenate((x1, x2, x3))
y=np.concatenate((y1, y2, y3))

df = pd.DataFrame({
    'x': x,
    'y': y
})

xmin=-16
ymin=-16
xmax=-xmin
ymax=-ymin

#seeds random
#np.random.seed(200)
k = 3
# centroids[i] = [x, y]
centroids = {
    i+1: [np.random.randint(xmin, xmax), np.random.randint(ymin, ymax)]
    for i in range(k)
}
    
fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color='k')
#need a color for each k centroid
colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'y', 5: 'c', 6: 'm'}
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.show()
#shows initial pts and centroids


## Assignment Stage

def assignment(df, centroids):
    for i in centroids.keys():
        # Euclidean dist: sqrt((x1 - x2)^2 + (y1 - y2)^2)
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df

df = assignment(df, centroids)
#print first 5 pts:
#print(df.head())

#prints initial clustering:
'''fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.show()'''


## Update Stage

import copy

old_centroids = copy.deepcopy(centroids)

def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return k

centroids = update(centroids)

#prints new centroid location:   
'''fig = plt.figure(figsize=(5, 5))
ax = plt.axes()
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
for i in old_centroids.keys():
    old_x = old_centroids[i][0]
    old_y = old_centroids[i][1]
    dx = (centroids[i][0] - old_centroids[i][0]) * 0.75
    dy = (centroids[i][1] - old_centroids[i][1]) * 0.75
    ax.arrow(old_x, old_y, dx, dy, head_width=2, head_length=3, fc=colmap[i], ec=colmap[i])
plt.show()'''


# Continue until all assigned categories don't change any more
while True:
    closest_centroids = df['closest'].copy(deep=True)
    centroids = update(centroids)
    df = assignment(df, centroids)
    if closest_centroids.equals(df['closest']):
        break

fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.show()