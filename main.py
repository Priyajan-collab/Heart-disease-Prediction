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

X_test=data_test.drop("DEATH_EVENT",axis=1)
Y_test=data_test.loc[:,["DEATH_EVENT"]]
col_head=X.columns.values
#converting to numpy
X=X.to_numpy()
Y=Y.to_numpy()
Y_test=Y_test.to_numpy()
X_test=X_test.to_numpy()

# flattening to (239,) from (239,1)
Y=Y.flatten()
Y_test = Y_test.flatten() 

X_mean=np.mean(X, axis=0)
X_std=np.std(X, axis=0)

X = (X - X_mean) / X_std

X_test = (X_test - X_mean) / X_std





fig, axes =pt.subplots(2,6,sharey=True,figsize=(5,8))

fig2,ax2=pt.subplots()

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




# initializing weights and bias
w = np.random.randn(X.shape[1]) * 0.01 
b=np.random.uniform()
lr=0.01
lambda_=0.001

class Logistic_Regression():
    def __init__ (self,x,y,w,b,lr,lambda_):
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
        epsilon = 1e-10
        self.cost_val=np.sum(-(self.y)*np.log(self.z+epsilon)+(1-self.y)*np.log(1-self.z+epsilon))/self.m
        return self.cost_val
    def regularaization(self):
        self.cost_reg=self.cost_val+(self.lambda_ / (2 * self.m)) * np.sum(self.w**2)     
        return self.cost_reg
    def update_weights_bias(self):
        d_w = np.dot((self.z - self.y),self.x) / self.m
        d_b=np.mean(self.z-self.y)
        
        # print(f"weights:{self.w} and bias{self.b}")
        self.w-=self.lr*d_w
        self.b-=self.lr*d_b
    def regularaization_update_weights_bias(self):
        d_w = np.dot((self.z - self.y),self.x) / self.m
        d_b=np.mean(self.z-self.y)
        self.w-= self.lr*d_w + self.lambda_ * self.w
        self.b-=self.lr*d_b    
    def predict_newdata(self,x_new):
        self.ys=np.dot(x_new,w)+self.b
        self.z=1/(1+np.exp(-self.ys))
        predictions = (self.z > 0.5).astype(int)
        return predictions
 
 
        
obj=Logistic_Regression(X,Y,w,b,lr,lambda_)
x_data=[]
y_data=[]
def train():
    for i in range(1000):
        obj.predict()
        obj.sigmoid()
        obj.cost()
        n=obj.regularaization()
        print(n)
        x_data.append(n)
        y_data.append(i)
        obj.update_weights_bias()
        # obj.regularaization_update_weights_bias()

train()
# print(X.shape,w.shape)
def cost_graph():
    ax2.scatter(x_data,y_data,c="r")
    ax2.set_title("Cost vs iteration") 
    ax2.set_ylabel("iteration")
    ax2.set_xlabel("cost")
        
        
vis_data()
cost_graph()

pre=obj.predict_newdata(X_test)
accuracy = np.mean(pre == Y_test) * 100
print(f"Accuracy: {accuracy:.2f}%")

x_random=X_test[15]
y_random=Y_test[15]

z=np.dot(x_random,obj.w)+obj.b
g=1/(1+np.exp(-z))

print(f"x:{x_random},y:{y_random},g:{g}")

pt.show()    