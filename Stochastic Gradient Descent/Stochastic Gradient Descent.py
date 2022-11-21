import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import sys

test_path=sys.argv[1]

X0=np.ones(1000000)
X1=np.random.normal(loc=3,scale=2,size=1000000)
X2=np.random.normal(loc=-1,scale=2,size=1000000)
X=np.vstack((X0,X1,X2))

noise=np.random.normal(loc=0,scale=2**0.5,size=1000000)
theta_original=np.array([3,1,2])
Y=np.dot(theta_original,X)+noise

alpha=0.001
theta=np.zeros(3)

def shuffle_2_array(x,y):
  i=np.arange(len(y))
  np.random.shuffle(i)
  return x[:,i],y[i]

X,Y=shuffle_2_array(X,Y)

def cost_function(X,Y,Q):
    cost=0
    m=X.shape[1]
    QX=np.dot(Q,X)
    error=QX-Y
    cost=np.dot(error,error)
    cost=(cost/(2*m))
    
    return cost
  

def gradient_function(X,Y,Q):
    m=X.shape[1]
    n=X.shape[0]
    #dj_dq=[0 for i in range(n)] ....this is for list
    dj_dq=np.zeros(n)
    QX=np.dot(Q,X)
    error=QX-Y
    dj_dq=np.dot(error,X.transpose())
    dj_dq=dj_dq/m

    return dj_dq

def total_batch(X,r):
  size=X.shape[1]//r
  return size

J_history=[]
theta_history=[]
theta_for_plot=[]

def SGD(r,checkpoint,X,Y,theta,delta):
  m=X.shape[1]
  n=X.shape[0]
  iter=0
  inner=False
  start=time.time()
  while True:
    for i in range(total_batch(X,r)):
      x_new=X[:,i*r:(i+1)*r]
      y_new=Y[i*r:(i+1)*r]
      dj_dq=gradient_function(x_new,y_new,theta)
      J_history.append(cost_function(x_new,y_new,theta))
      theta=theta-alpha*dj_dq
      theta_for_plot.append(theta)
      iter+=1
      if (iter)%checkpoint==0:
        theta_history.append(theta)
        #print(theta)
        theta_array=np.array(theta_history) 
        if theta_array.shape[0]>=2 and (abs(theta_array[-1]-theta_array[-2])<delta).all():
          inner=True
          break
    if inner:
      break      
  final=time.time()-start
  theta_for_plot2=np.array(theta_for_plot)
  #print(iter)
  return (theta,final,theta_for_plot2)   

Q1,t1,theta_plot1=SGD(100,1000,X,Y,theta,1e-02)
#print(Q1)
#print(t1)

def plotting(Q):
  fig=plt.figure()
  ax=fig.add_subplot(projection='3d')
  theta_movement=ax.plot(Q[:,0],Q[:,1],Q[:,2],color='blue')
  ax.set_xlabel('Theta 0')
  ax.set_ylabel('Theta 1')
  ax.set_zlabel('Theta 2')
  ax.set_xlim(-0.5,3.5)
  ax.set_ylim(-0.5,1.5)
  ax.set_zlim(-0.5,2.5)
  plt.show()

#plotting(theta_plot1)

database=np.array(pd.read_csv((test_path + "\X.csv"),header=None))
X_test=database[:,0:2]
#Y_test=database[:,2]
X0=np.ones((X_test.shape[0],1))
X_test=np.hstack((X0,X_test))
X_test=X_test.transpose()

#test_error=cost_function(X_test,Y_test,Q1)
prediction=np.dot(Q1,X_test)

with open('result_2.txt','w') as f:
    for i in range (prediction.shape[0]):
        f.write(str(prediction[i]))
        f.write("\n")
    f.close()

#print(test_error)

#error_wrt_original=cost_function(X_test,Y_test,theta_original)
#print(error_wrt_original)