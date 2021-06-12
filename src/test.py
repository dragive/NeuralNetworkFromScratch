import numpy as np

from NN import *
import json
model = Model()

performance = []
model = Model()

model.addLayer(4,ActivationFunctions.unipolar)
model.addLayer(4,ActivationFunctions.unipolar)
model.addLayer(1,ActivationFunctions.unipolar)

def monitor(iteration:int = None,wholeError:float=0.0,correction = None):
    global performance
    performance.append(wholeError)
model.iterations = 200
model.learningFactor=0.5
model.biasState= True
    
model.compileModel()

model.setExternalMonitor(monitor=monitor)
train_dataset= {
    'egzo': [[1,1,0,0],[1,0,0,0],[1,1,1,1]], 
    'endo':[ [1],      [0],      [0.5]]
}


print(json.dumps(train_dataset))
model.upload_train_dataset(train_dataset)
model.upload_test_dataset({
    'egzo': [], 
    'endo': []
})
print(model.forward([1,0,0,0]))
print(model.forward([1,1,0,0]))
print(model.forward([1,1,1,1]))

one=model.train()
one

print("0         ",model.forward([1,0,0,0]))
print("1         ",model.forward([1,1,0,0]))
print("0.5       ",model.forward([1,1,1,1]),end="\n\n")
print("evaluate: ",model.evaluate())

import matplotlib.pyplot as plt
plt.plot(performance)
plt.show()
np.min(performance)