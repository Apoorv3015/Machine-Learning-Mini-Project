import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

train_path=sys.argv[1]
test_path=sys.argv[2]

#database_x=pd.read_csv(r"C:\Users\apoor\OneDrive\Documents\ML_Assignment\Assignment_1\Q1\q4x.dat", header=None,delimiter="  ",engine='python')
database_x=pd.read_csv((train_path+"\X.csv"), header=None)
X=np.array(database_x)

#database_y=pd.read_csv(r"C:\Users\apoor\OneDrive\Documents\ML_Assignment\Assignment_1\Q1\q4y.dat", header=None)
database_y=pd.read_csv((train_path+"\Y.csv"), header=None)
Y=np.array(database_y)
Y=np.array([0 if i=='Alaska' else 1 for i in Y])

test_X=np.array(pd.read_csv((test_path+"\X.csv"), header=None))


Mean=np.mean(X,axis=0,keepdims=True)
std_dev=np.std(X,axis=0,keepdims=True)
X_normalized=(X-Mean)/std_dev

X_test_normalized=(test_X-Mean)/std_dev

X_for_plot=X_normalized.transpose()

def phi(Y):
  count=np.count_nonzero(Y==1)
  length=Y.shape[0]
  return count/length

p=phi(Y)
#print(p)

def meu0(X,Y):
  denominator=np.count_nonzero(Y==0)
  i=0
  sum=np.zeros((1,2))
  for x in Y:
    if x==0:
      sum+=X[i,:]
      i+=1
    else: 
      i+=1
  return sum/denominator

m0=meu0(X_normalized,Y)
#print(m0)

def meu1(X,Y):
  denominator=np.count_nonzero(Y==1)
  i=0
  sum=np.zeros((1,2))
  for x in Y:
    if x==1:
      sum+=X[i,:]
      i+=1
    else: 
      i+=1
  return sum/denominator

m1=meu1(X_normalized,Y)
#print(m1)

def sigma(X,Y,mue0,mue1):                #sigma is covariance matrix, i.e. same for both the classes
  x=np.copy(X)
  i=0
  for y in Y:
    if y==0:
      x[i]=X[i]-mue0
      i+=1
    else:
      x[i]=X[i]-mue1
      i+=1
  numerator=np.dot(x.transpose(),x)
  length=X.shape[0]
  return numerator/length

s=sigma(X_normalized,Y,m0,m1)
#print(s)

def linear_decision_boundary(phi,mue0,mue1,sigma):
  m0_sigma_m0=np.dot(np.dot(mue0,np.linalg.inv(sigma)),mue0.transpose())
  #print(m0_sigma_m0)
  m1_sigma_m1=np.dot(np.dot(mue1,np.linalg.inv(sigma)),mue1.transpose())
  #print(m1_sigma_m1)
  logterm=np.log(phi/(1-phi))
  #print(logterm)
  constant_term=logterm-0.5*(-m0_sigma_m0+m1_sigma_m1)
  #print(constant_term)
  sigma_mue_matrix=np.dot(mue0,np.linalg.inv(sigma))-np.dot(mue1,np.linalg.inv(sigma))    # 1x2 matrix generated 
  #print(sigma_mue_matrix)
  x1=np.linspace(-2,2,50)
  x2=(constant_term[0][0]/sigma_mue_matrix[0][1])-(sigma_mue_matrix[0][0]/sigma_mue_matrix[0][1])*x1
  return x1,x2


x1_for_plot,x2_for_plot=linear_decision_boundary(p,m0,m1,s)


def plotting(x,y):
  X_positive=X_for_plot[:,Y==1]
  X_negative=X_for_plot[:,Y==0]

  fig=plt.figure()
  ax=fig.add_subplot()
  ax.scatter(X_positive[0],X_positive[1],color='blue',marker='o',label='Canada')
  ax.scatter(X_negative[0],X_negative[1],color='red',marker='*',label='Alaska')
  ax.set_xlabel('X1')
  ax.set_ylabel('X2')

  ax.plot(x,y,c='black',label='Linear decision boundary')
  ax.legend()
  plt.show()

#plotting(x1_for_plot,x2_for_plot)

def sigma0(mue0,X,Y):
  x=np.copy(X)
  i=0
  for y in Y:
    if y==0:
      x[i]=X[i]-mue0
      i+=1
    else:
      x[i]=0
      i+=1
  numerator=np.dot(x.transpose(),x)
  denominator=np.count_nonzero(Y==0)
  return numerator/denominator

def sigma1(mue1,X,Y):
  x=np.copy(X)
  i=0
  for y in Y:
    if y==0:
      x[i]=0
      i+=1
    else:
      x[i]=X[i]-mue1
      i+=1
  numerator=np.dot(x.transpose(),x)
  denominator=np.count_nonzero(Y==1)
  return numerator/denominator

s0=sigma0(m0,X_normalized,Y)
s1=sigma1(m1,X_normalized,Y)
#print(s0)
#print(s1)

def quadratic_decision_boundary(phi,mue0,mue1,sigma0,sigma1):
  m0_sigma0_m0=np.dot(np.dot(mue0,np.linalg.inv(sigma0)),mue0.transpose())
  #print(m0_sigma0_m0)
  m1_sigma1_m1=np.dot(np.dot(mue1,np.linalg.inv(sigma1)),mue1.transpose())
  #print(m1_sigma1_m1)
  logterm=np.log(phi/(1-phi))
  #print(logterm)
  constant_term=logterm-0.5*(-m0_sigma0_m0+m1_sigma1_m1)+0.5*np.log(np.linalg.det(sigma0)/np.linalg.det(sigma1))  # Constant
  #print(constant_term)
  sigma_mue_matrix=np.dot(mue1,np.linalg.inv(sigma1))-np.dot(mue0,np.linalg.inv(sigma0))    # Matrix M, 1x2 shape
  #print(sigma_mue_matrix)
  sigma_matrix=0.5*(np.linalg.inv(sigma0)-np.linalg.inv(sigma1))  # Matrix A
  #print(sigma_matrix)

  # Equation --> (A[0][0])X1^2 + (A[1][1])X2^2 + (A[1][0]+A[0][1])X1X2 + (M[0][0])X1 + (M[0][1])X2 + Constant
  # (A[1][1])X2^2 + [ (A[1][0]+A[0][1])X1 + (M[0][1]) ]X2 + [ (A[0][0])X1^2 + (M[0][0])X1 + Constant ]  
  #   a X2^2 + b X2 + c
  
  x1=np.linspace(-2,2,50)
  a= sigma_matrix[1][1]
  b= (sigma_matrix[1][0]+sigma_matrix[0][1])*x1 + sigma_mue_matrix[0][1] 
  c= sigma_matrix[0][0]*(x1**2) + sigma_mue_matrix[0][0]*x1 + constant_term[0][0]
  x2= ((-b)-np.sqrt(b**2 - 4*a*c))/(2*a)
  return x1,x2

x3_for_plot,x4_for_plot=quadratic_decision_boundary(p,m0,m1,s0,s1)

def plotting(x,y,x1,y1):
  X_positive=X_for_plot[:,Y==1]
  X_negative=X_for_plot[:,Y==0]

  fig=plt.figure()
  ax=fig.add_subplot()
  ax.scatter(X_positive[0],X_positive[1],color='blue',marker='o',label='Canada')
  ax.scatter(X_negative[0],X_negative[1],color='red',marker='*',label='Alaska')
  ax.set_xlabel('X1')
  ax.set_ylabel('X2')

  ax.plot(x,y,c='black',label='Linear decision boundary')
  ax.plot(x1,y1,c='green',label='Quadratic decision boundary')
  ax.legend()
  plt.show()
  
#plotting(x1_for_plot,x2_for_plot,x3_for_plot,x4_for_plot)

def test_result(X,phi,mue0,mue1,sigma0,sigma1):
  m0_sigma0_m0=np.dot(np.dot(mue0,np.linalg.inv(sigma0)),mue0.transpose())
  m1_sigma1_m1=np.dot(np.dot(mue1,np.linalg.inv(sigma1)),mue1.transpose())
  logterm=np.log(phi/(1-phi))
  constant_term=logterm-0.5*(-m0_sigma0_m0+m1_sigma1_m1)+0.5*np.log(np.linalg.det(sigma0)/np.linalg.det(sigma1))  # Constant
  M=np.dot(mue1,np.linalg.inv(sigma1))-np.dot(mue0,np.linalg.inv(sigma0))    # Matrix M, 1x2 shape
  A=0.5*(np.linalg.inv(sigma0)-np.linalg.inv(sigma1))
  # Equation --> (A[0][0])X1^2 + (A[1][1])X2^2 + (A[1][0]+A[0][1])X1X2 + (M[0][0])X1 + (M[0][1])X2 + Constant
  prediction=[]
  for i in range(X.shape[0]):
    value=(A[0][0]*X[i][0]*X[i][0] + A[1][1]*X[i][1]*X[i][1]+(A[1][0]+A[0][1])*X[i][0]*X[i][1] + M[0][0]*X[i][0]+M[0][1]*X[i][1])
    if value>0:
      prediction.append('Canada')
    else: 
      prediction.append('Alaska')
  with open('result_4.txt','w') as f:
    for i in prediction: 
      f.write(i)
      f.write("\n")
    f.close()

test_result(X_test_normalized,p,m0,m1,s0,s1)

def test_data_plotting(X,x,y,x1,y1):
  X=X.transpose()
  fig=plt.figure()
  ax=fig.add_subplot()
  ax.scatter(X[0],X[1],color='red',marker='*')
  ax.set_xlabel('X1')
  ax.set_ylabel('X2')
  ax.plot(x,y,c='black',label='Linear decision boundary')
  ax.plot(x1,y1,c='green',label='Quadratic decision boundary')
  ax.legend()
  plt.show()
  
#test_data_plotting(X_test_normalized,x1_for_plot,x2_for_plot,x3_for_plot,x4_for_plot)