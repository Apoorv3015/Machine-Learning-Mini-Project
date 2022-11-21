import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

training_path=sys.argv[1]
test_path=sys.argv[2]

database_x=pd.read_csv((training_path+"\X.csv"), header=None)
X=np.array(database_x)

database_y=pd.read_csv((training_path+"\Y.csv"), header=None)
Y=np.array(database_y)

test_X=np.array(pd.read_csv((test_path+"\X.csv"), header=None))


Mean=np.mean(X,axis=0,keepdims=True)
std_dev=np.std(X,axis=0,keepdims=True)
X_normalized=(X-Mean)/std_dev
X0=np.ones((X.shape[0],1))
X_normalized=np.hstack((X0,X_normalized))

X_test_normalized=(test_X-Mean)/std_dev
X0_test=np.ones((test_X.shape[0],1))
X_test_normalized=np.hstack((X0_test,X_test_normalized))

X_for_plot=X_normalized.transpose()

Y=Y.transpose()
X_normalized=X_normalized.transpose()


theta=np.zeros((1,3))

def sigmoid(Q,X):
  QX=np.dot(Q,X)
  return 1/(1+np.exp(-QX))

def log_likelihood(Q,X,Y):
  term1=np.dot(np.log(sigmoid(Q,X)),Y.transpose())[0][0]
  term2=np.dot(np.log(1-sigmoid(Q,X)),(1-Y).transpose())[0][0]
  return term1+term2

def gradient(Q,X,Y):
  error=Y-sigmoid(Q,X)
  dl_dq=np.dot(error,X.transpose())
  return dl_dq


def hessian(Q,X,Y):
  prod=sigmoid(Q,X)*(1-sigmoid(Q,X))
  prod=np.diag(prod[0])
  hessian_matrix=-np.dot(np.dot(X,prod),X.transpose())
  return hessian_matrix


logl_history=[]
logl_history.append(log_likelihood(theta,X_normalized,Y))
iter=0
while True:
  dl_dq=gradient(theta,X_normalized,Y)
  H=hessian(theta,X_normalized,Y)
  H_inverse=np.linalg.inv(H)
  iter+=1
  theta=theta-np.dot(dl_dq,H_inverse)
  logl_history.append(log_likelihood(theta,X_normalized,Y))
  if abs(logl_history[-2]-logl_history[-1])<1e-10:
    break

#print(logl_history)
#print(iter)
#print(theta)

def test_result(X,Q):
  h_theta=sigmoid(Q,X.transpose())

  with open('result_3.txt','w') as f:
    for i in range(h_theta.shape[1]): 
      if h_theta[0][i]>=0.5:
        f.write(str(1))
        f.write("\n")
      else: 
        f.write(str(0))
        f.write("\n")
    f.close()

test_result(X_test_normalized,theta)

def plotting(theta):
  X_positive=X_for_plot[1:,Y[0,:]==1]
  X_negative=X_for_plot[1:,Y[0,:]==0]

  fig=plt.figure()
  ax=fig.add_subplot()
  ax.scatter(X_positive[0],X_positive[1],color='blue',marker='o',label='Class 1')
  ax.scatter(X_negative[0],X_negative[1],color='red',marker='*',label='Class 0')
  ax.set_xlabel('X1')
  ax.set_ylabel('X2')
  Y_for_plot=-(theta[0][1]/theta[0][2])*X_for_plot[1]-(theta[0][0]/theta[0][2])
  ax.plot(X_for_plot[1],Y_for_plot,c='black',label='Boundary line')
  ax.legend()
  plt.show()

#plotting(theta)