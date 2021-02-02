import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# generate x values to plot
x_plot = np.linspace(-5, 5, 1000)

#gen data
"""
pi=math.pi
n1=10
x=np.linspace(0,1,n1).reshape(1,-1)
y=np.sin(2*pi*x).reshape(1,-1)
x_original=x.copy()
rng = np.random.RandomState(0)
rng.shuffle(x)
y = f(x)

print(x)
print(y)
"""

xx=np.linspace(0,1,10)
#print(xx)

# input data
x = np.array(xx)
y = np.array([.3123, .9111, .9912, .9874, .0545, .0642, -.9132, -.5456, -.6123, .2112]) #randomly generated from code above. I hard printed these values to prevent regenerating new numbers each run.
#print(x)
#print(y)

def f(x):
    """ function to approximate by polynomial interpolation"""
    return np.sin(2*math.pi*x)


# create matrix versions of these arrays
X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]

colors = ['teal', 'yellowgreen', 'gold']
lw = 2

plt.plot(x_plot, f(x_plot), color='red', linewidth=lw,
         label="actual function")
plt.scatter(x, y, color='navy', s=30, marker='o', label="training points")

#change lamda value here
a=0
m=9
for count, degree in enumerate([m]): #3 max
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=a,)) #insert lamda
    z=model.fit(X, y)
    y_plot = model.predict(X_plot)
    plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw, label="M=%d" % degree)
    
z = np.polyfit(x_plot, y_plot, m)
print("coefs:", z) #starts with highest power coef
  

plt.xlabel('')
plt.ylabel('')
plt.title('lamda=%f' % a)
plt.legend(loc='lower left')
plt.ylim(-4,4)
plt.xlim(-0.1,1.1)
plt.show()