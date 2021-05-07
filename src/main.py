import numpy as np
from math import e as E
np.random.seed(0)

# training dataset
X = [[0  ,1  ,2  ,3  ]
    ,[1  ,0  ,2.5,3.4]
    ,[0  ,-1 ,2.5,3.2]
]
# initializing biases as 0 
# but weights from -1 up to 1
# why? imagine 2*2*2*...*2 through all layers
# it becomes giantic
class Layer:
    def __init__(self,n_inputs,n_neurons):
        self.weights = np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

def softmax(array:np.array):
    temp = np.exp(array)
    return temp / np.sum(temp,axis = 1,keepdims = True) #

def relu(array:np.array): return np.maximum(array,0)

print(softmax(np.array(X)))



layer1 = Layer(4,5)
layer2 = Layer(5,2)
layer1.forward(X)
layer2.forward(layer1.output)
print(layer2.output)
