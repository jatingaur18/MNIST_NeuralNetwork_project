import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('mnist_test.csv')
# print(data.head())
data = np.array(data)
m,n =data.shape

np.random.shuffle(data)

data_dev = data[0:1000].T
y_dev = data_dev[0]
x_dev = data_dev[1:n]
x_dev = x_dev / 255.

data_train = data[1000:m].T
y_train = data_train[0]
x_train = data_train[1:n]
x_train = x_train / 255.
_,m_train = x_train.shape

def init_params():
    W1 =np.random.rand(10, 784) - 0.5
    b1 =np.random.rand(10, 1) - 0.5
    W2 =np.random.rand(10, 10) - 0.5
    b2 =np.random.rand(10, 1) - 0.5
    return W1,b1,W2,b2

def relu(z):
    return np.maximum(0,z)

def softmax(z):
    return np.exp(z)/ sum(np.exp(z))

def forward_prop(W1,b1,W2,b2,X):
    Z1=W1.dot(X) +b1
    A1 =relu(Z1)
    Z2=W2.dot(A1) +b2
    A2 =softmax(Z2)
    return Z1, A1, Z2, A2
 
def one_hot(y):
    one_hot_y= np.zeros((y.size, y.max()+1))
    one_hot_y[np.arange(y.size),y]=1
    one_hot_y=one_hot_y.T
    return one_hot_y

def deriv_ReLU(z):
    return z>0

def back_prop(Z1,A1,Z2,A2,W2,X,Y):
    one_hot_Y=one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m * dZ2.dot(A1.T)
    dB2 = 1/m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2)*deriv_ReLU(Z1)
    dW1 = 1/m * dZ1.dot(X.T)
    dB1 = 1/m * np.sum(dZ1)
    return dW1,dB1,dW2,dB2

def update_params(W1,b1,W2,b2,dW1,dB1,dW2,dB2,alpha):
    W1 =W1 -alpha*dW1
    b1 =b1 -alpha*dB1
    W2 =W2 -alpha*dW2
    b2 =b2 -alpha*dB2
    return W1,b1,W2,b2

def get_prediction(A2):
    return np.argmax(A2,0)

def get_accuracy(predictions,Y):
    print(predictions,Y)
    return np.sum(predictions == Y)/Y.size

def gradiant_descent(X,Y,alpha,iterations):
    W1,b1,W2,b2 = init_params()
    for i in range(iterations):
        Z1,A1,Z2,A2 = forward_prop(W1,b1,W2,b2,X)
        dW1,db1,dW2,db2 = back_prop(Z1,A1,Z2,A2,W2,X,Y)
        W1,b1,W2,b2 = update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,alpha)
        if i % 10 == 0:
            print("iteration: ",i)
            predictions = get_prediction(A2)
            print("accuracy: ",get_accuracy(predictions,Y))
    return W1,b1,W2,b2
   
W1,b1,W2,b2 = gradiant_descent(x_train,y_train,0.10,500)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_prediction(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = x_train[:, index, None]
    prediction = make_predictions(x_train[:, index, None], W1, b1, W2, b2)
    label = y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    
# test_prediction(0, W1, b1, W2, b2)
# test_prediction(1, W1, b1, W2, b2)
# test_prediction(2, W1, b1, W2, b2)
# test_prediction(3, W1, b1, W2, b2)

dev_predictions = make_predictions(x_dev, W1, b1, W2, b2)
print(get_accuracy(dev_predictions, y_dev))