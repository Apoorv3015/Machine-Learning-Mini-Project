import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time


training_path=sys.argv[1]
test_path=sys.argv[2]

database_x=pd.read_csv((training_path+"\X.csv"), header=None)        # read X for training 
X=np.array(database_x)

database_y=pd.read_csv((training_path+"\Y.csv"), header=None)          # read Y for training 
Y=np.array(database_y)

test_X=np.array(pd.read_csv((test_path+"\X.csv"), header=None))      # read X for testing 

                                          
Mean=np.mean(X,axis=0,keepdims=True)
std_dev=np.std(X,axis=0,keepdims=True)
X_normalized=(X-Mean)/std_dev
X0=np.ones(X.shape)
X_normalized=np.hstack((X0,X_normalized))          # M x N shape

                                 
X_test_normalized=(test_X-Mean)/std_dev
X0_test=np.ones(test_X.shape)
X_test_normalized=np.hstack((X0_test,X_test_normalized))      #test data normalize

theta=np.zeros((X_normalized.shape[1],1))               # N x 1 shape

def cost_function(X,Y,Q):
    cost=0
    m=X.shape[0]
    XQ=np.dot(X,Q)
    error=XQ-Y
    cost=np.dot(error.transpose(),error)[0][0]
    cost=(cost/(2*m))
    return cost

def gradient_function(X,Y,Q):
    m=X.shape[0]
    n=X.shape[1]
    dj_dq=np.zeros(n)
    XQ=np.dot(X,Q)
    error=XQ-Y

    for j in range(n):
        x_=X[:,j].reshape(1,m)
        #xnew=x_.reshape(m,1)
        #print(x_.shape)
        dj_dq[j]=np.dot(x_,error)[0][0]
        dj_dq[j]=dj_dq[j]/m

    return dj_dq

JQ_history=[]
theta0_history=[]
theta1_history=[]
JQ_history.append(cost_function(X_normalized,Y,theta))
theta0_history.append(theta[0][0])
theta1_history.append(theta[1][0])

alpha=0.01           # Default alpha=0.01, taken by me.
i=0
start=time.time()
while True:
    n=X_normalized.shape[1]
    m=X_normalized.shape[0]
    dj_dq=gradient_function(X_normalized,Y,theta)
    
    for j in range(n):
        theta[j]=theta[j]-alpha*dj_dq[j]
    
    theta0_history.append(theta[0][0])
    theta1_history.append(theta[1][0])
    i+=1
    JQ_history.append(cost_function(X_normalized,Y,theta))
    if abs(JQ_history[-1]-JQ_history[-2])<=1e-12:
        break

end=time.time()
#print(theta)
#print(i)
#print(end-start)

def test_result(X,Q):
    predictions=np.dot(X,Q)
    with open('result_1.txt','w') as f:
        for i in range (predictions.shape[0]):
            f.write(str(predictions[i][0]))
            f.write("\n")
        f.close() 

test_result(X_test_normalized,theta)

#All Plotting functions below.......

def plot_Hypothesis_line():
    plt.scatter(X_normalized[:,1],Y ,marker='x',c='r',label='Training data')
    plt.xlabel('Acidity')
    plt.ylabel('Density')
    X_plot=np.arange(-2,6,1)
    Y_plot=X_plot*theta[1]+theta[0]
    plt.plot(X_plot,Y_plot,label='Hypothesis')
    plt.legend()
    plt.show()

#plot_Hypothesis_line()

def cost_fun_3d(x,y,Q0,Q1,n):
  m=x.shape[0]
  cost=np.zeros((n,n))
  for i in range(m):
    cost+=(Q0+Q1*x[i][1]-y[i][0])**2
  cost=cost/(2*m)
  return cost

def surface_plot():
    fig=plt.figure()
    ax = fig.add_subplot(projection ='3d')
    Q0_3d,Q1_3d=np.meshgrid(np.linspace(0,2,100),np.linspace(-1,1,100))
    Z_3d=cost_fun_3d(X_normalized,Y,Q0_3d,Q1_3d,100)
    ax.plot_surface(Q0_3d,Q1_3d,Z_3d, cmap ='viridis',alpha=0.8)
    ax.plot(theta0_history,theta1_history,JQ_history,color='black',markersize=10,lw=2)
    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')
    ax.set_zlabel('Cost')
    ax.set_title('3D plot of cost function')
    plt.show()

#surface_plot()

def contour():
    fig=plt.figure()
    ax = fig.add_subplot()
    Q0_3d,Q1_3d=np.meshgrid(np.linspace(0,2,100),np.linspace(-1,1,100))
    Z_3d=cost_fun_3d(X_normalized,Y,Q0_3d,Q1_3d,100)
    ax.contour(Q0_3d,Q1_3d, Z_3d, cmap = cm.RdYlBu,alpha=0.8)
    ax.plot(theta0_history,theta1_history,color='black')
    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')
    plt.show()
#contour()


