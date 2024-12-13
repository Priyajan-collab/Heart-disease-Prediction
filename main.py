import pandas as pd
import numpy as np
from matplotlib import pyplot as pt


data=pd.read_csv("dataset/heart_failure_clinical_records_dataset.csv")
data_train=data.sample(frac=0.8,random_state=1639)
data_test=data.drop(data_train.index)

# preliminary data cleaning
# print(data.isnull().sum()) #checking if data is null or not
# print(data.isna().sum()) #to check if the data is null

# checkingd data types
# print(data.dtypes)

# visualize data
X=data_train.drop("DEATH_EVENT",axis=1)
Y=data_train.loc[:,["DEATH_EVENT"]]

col_head=X.columns.values
#converting to numpy
X=X.to_numpy()
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
Y=Y.to_numpy()

fig, axes =pt.subplots(2,6,sharey=True,figsize=(5,8))


def vis_data():
    row=0
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):  # Enumerate over the axes array
            if (i==1):
                row=6
            
            axes[i][j].scatter(X[:,row+j], Y, c="r", s=1)  # Replace X[:, i] with your data
            axes[i][j].set_title(f"Feature {row+j+1}")
            axes[i][j].set_xlabel(f" {col_head[j+row]} ")
            axes[i][j].set_ylabel("Death")


# vis_data()

# initializing weights and bias
w=np.zeros(X.shape[1])
b=np.random.uniform()

class Logistic_Regression():
    def __init__ (x,y,w,b,lr,lambda_,self):
        self.x=x
        self.y=y
        self.w=w
        self.b=b
        self.lr=lr
        self.lambda_=lambda_
        self.m=len(y)
    def predict(self):
        self.ys=np.dot(self.x,w)+self.b
    def sigmoid(self):
        self.z=1/(1+np.exp(-self.ys))
    def cost(self):
        self.cost_val=np.sum(-(self.y)*np.log(self.z)+(1-self.y)*np.log(1-self.z))/self.m
        return self.cost
    def regularaization(self):
        self.cost_reg=self.cost_val + np.mean(self.lambda_*self.w)     
        return self.cost_reg
    def update_weights_bias(self):
        d_w=np.mean(np.dot(np.sum(self.z-self.y),self.x))
        d_b=np.mean(np.sum(self.z-self.y))
        self.w+=d_w
        self.b+=d_b
    def regularaization_update_weights_bias(self):
        d_w=np.mean(np.dot(np.sum(self.z-self.y),self.x))
        d_b=np.mean(np.sum(self.z-self.y))
        self.w+=d_w+np.mean(self.lambda_*self.w)
        self.b+=d_b    
      
        
           
    
        
        
        



pt.show()    