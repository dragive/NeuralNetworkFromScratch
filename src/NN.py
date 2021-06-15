#!/usr/bin/python3
import numpy as np


class ActivationFunctions:
        
    def Export(func):
        return "uni" if func == ActivationFunctions.unipolar or func == ActivationFunctions.Default else 'bi'
    def Import(name):
        return ActivationFunctions.bipolar if name == 'bi' else ActivationFunctions.unipolar

    def unipolar(data,derivative = False):
        if derivative:
                   
            return (1-data)*data
        
        temp = np.array(data)
        return 1.0/(1.0+np.exp(-temp))
    
    def bipolar(data,derivative = False):
        
        if derivative:
            return 1-(data*data)     

        return -(1.0-np.exp(np.array(data)))/(1.0+np.exp(np.array(data)))

    def Default(data,derivative=False):
        return ActivationFunctions.unipolar(data,derivative)

class LossFunctions:
    def Export(func):
        return "AE"

    def Import(str):
        return LossFunctions.AE

    def Default(estimator,goal,derivative=False):
        return LossFunctions.AE(estimator,goal,derivative)
    def AE(estimator,goal,derivative=False):
        
        if derivative:
            return estimator-goal
        return np.power((estimator-goal),2)/2
        
        

class Layer:


    def __init__(self,shape,function=ActivationFunctions.Default) -> None:
        assert len(shape)==2
        assert isinstance(shape[0],int) and isinstance(shape[1],int)
        assert shape[0]>0 and shape[1]>0
        self.function=function
        self.shape = shape

        self._init_matrix()
        self.output=None
        self.input=None

   
    def __init__(self,shape_x:int,shape_y:int,function=ActivationFunctions.Default) -> None:
        assert shape_y>0 and shape_x>0

        self.shape = (shape_x,shape_y)
        self.function=function
        self._init_matrix()
        self.output=None
        self.input=None

    def _init_matrix(self):
        self.matrix = (np.random.randn(self.shape[0],self.shape[1]))*2-1
        self.bias=np.random.randn(self.shape[1])

    def Export(self):
        ret = {}
        i=0
        ret['shape'] = self.shape
        # print(ret['shape']);i+=1
        ret['activationFunction'] = ActivationFunctions.Export(self.function)
       
        ret['output'] = self.output.tolist() if self.output is not None else None
        # print(ret['output'] );i+=1
        ret['input'] = np.array(self.input).tolist() if self.input is not None else None
        # print(ret['input']);i+=1
        ret['matrix'] = self.matrix.tolist() if self.matrix is not None else None
        # print(ret['matrix']);i+=1
        ret['bias'] = self.bias.tolist() if self.bias is not None else None
        # print(ret['bias'] );i+=1

        return ret

    def Import(self,imp):
        assert isinstance(imp,dict)
        self.shape = imp['shape']
        self.function = ActivationFunctions.Import( imp['activationFunction'] )
        self.output = np.array(imp['output']) if imp['output'] != None else None 
        self.input = np.array(imp['input'])if imp['input'] != None else None 
        self.matrix = np.array(imp['matrix'])if imp['matrix'] != None else None 
        self.bias = np.array(imp['bias'])if imp['bias'] != None else None 
        return self
            
    def forward(self,data,function=None,biasState = True):
        
        if function in [None]:
            function = self.function

        self.input=data.copy()
        temp=np.dot(data,self.matrix)
        temp_shape = temp.shape
        
        if biasState:
            self.dot = np.reshape(temp.copy()-self.bias,temp_shape)
        else:
            self.dot = temp.copy()

#             print(self.dot,"####",)
        self.output = function(self.dot)
        return self.output.copy()
    
    
    
    
    
    

class Model:

    def __init__(self):
        self.layers=[] # tymczasowe wartości, które potem są wykorzystywne do kompilacji modelu
        self.structure=[] # zbiór obiektów Layer
        self.learningFactor=0.2
        self.dataset={}
        self.loss_v=None
        self.iterations = 100
        self.biasState = True
        self.externalMonitor = None

    def Export(self):
        ret = {}

        ret['learningFactor']=self.learningFactor
        ret['layers']=[[x[0],ActivationFunctions.Export(x[1])] for x in self.layers]
        ret['structure'] = [i.Export() for i in self.structure] if self.structure != None else None
        ret['dataset'] = dict(self.dataset)

        ret['biasState'] = self.biasState
        ret['iterations'] = self.iterations
        print(ret)
        
        return ret



    def Import(self,imp):
        assert isinstance(imp,dict)

        self.learningFactor = imp['learningFactor']
        self.layers=[(i[0],ActivationFunctions.Import(i[1])) for i in imp['layers']] 
        self.structure = [Layer(1000,1000).Import(i) for i in imp['structure']] if imp['structure'] != None else None
        print(self.structure)
        self.dataset = imp['dataset']
        # self.loss_v = imp['loss_v']
        self.biasState = imp['biasState']
        self.iterations = self.iterations
 
    def addLayer(self,numberOfNeurons:int,function=ActivationFunctions.Default):
        assert numberOfNeurons>0
        # TODO to validate function
        self.layers.append((numberOfNeurons,function))
        
    def compileModel(self):
        assert len(self.layers)>=2
        self.structure=[]
        for i in range(len(self.layers)-1):
            self.structure.append(Layer(self.layers[i][0],self.layers[i+1][0],self.layers[i+1][1]))
            
    def forward(self,data):
        assert len(self.structure) >=1
        output = data
        for i in self.structure:
            output = i.forward(output,biasState=self.biasState)#.copy()
        return output#.copy()
    
    def upload_test_dataset(self,data): #TODO add validation
        self.dataset['test'] = data
        
    def upload_train_dataset(self,data): #TODO add validation
        self.dataset['train'] = data
    
    def loss(self,goal,function=ActivationFunctions.Default):
        return function(self.structure[-1].output,goal)
    
    def evaluate(self):
        return self.evaluate_train()
    def evaluate_test(self):
        suma = 0.0
        for x,y in zip(self.dataset['test']['egzo'],self.dataset['test']['endo']):
            temp = (self.forward(x)-y)
            temp *= temp
            suma +=temp
        return suma
    def evaluate_train(self):
        suma = 0.0
        for x,y in zip(self.dataset['train']['egzo'],self.dataset['train']['endo']):
            temp = (self.forward(x)-y)
            temp *= temp
            suma +=temp
        return suma
        
    def SGD(self,x,y):
        self.forward(x)
        #self.loss_v=self.loss(y,LossFunctions.AE)

        nabla_w = [0]*len(self.structure)
        nabla_b = [0]*len(self.structure)

        delta = LossFunctions.AE(self.structure[-1].output,y,derivative=True) \
                * self.structure[-1].function(self.structure[-1].output,derivative=True)
        
        self.structure[-1].input = np.expand_dims(self.structure[-1].input, axis=1)
        
        delta = np.expand_dims(delta, axis=1) 
        nabla_b[-1] = delta.copy()
        nabla_w[-1]= np.dot( self.structure[-1].input,delta.T)
        
        
        for l in range(2,len(self.structure)+1):
            
            z=None
            
            sp=self.structure[-l].function(self.structure[-l].output,derivative=True)

            delta = np.dot(delta,self.structure[-l+1].matrix.T)*sp
            nabla_b[-l] = delta.copy()

            self.structure[-l].input = np.expand_dims(self.structure[-l].input, axis=1)

            nabla_w[-l] = np.dot(self.structure[-l].input,delta)

        return nabla_w , nabla_b

    def train(self):

        counter = self.iterations
        while counter>=0:
            counter-=1

            arr = None
            for x,y in zip(self.dataset['train']['egzo'],self.dataset['train']['endo']):

                nabla_w, nabla_b = self.SGD(x,y)

                for i  in range(len(self.structure)):

                    self.structure[i].matrix = self.structure[i].matrix - np.multiply(self.learningFactor,nabla_w[i])
                    if self.biasState: 
                        self.structure[i].bias = self.structure[i].bias + self.learningFactor*nabla_b[i]
            perform_train = self.evaluate()
            perform_test = self.evaluate_test()
            
            self.internalMonitor(counter,perform_train,perform_test,(nabla_w.copy(),nabla_b.copy()))
    
    def setExternalMonitor(self, monitor=None):
        if not monitor == None:
            self.externalMonitor = monitor

    def internalMonitor(self,iteration:int = None,wholeError_train:float=0.0,wholeError_test:float=0.0,correction = None):
        if not self.externalMonitor == None:
            self.externalMonitor(iteration,wholeError_train,wholeError_test)
    
    def toggleBias(self,biasState:bool=True):
        self.biasState = biasState