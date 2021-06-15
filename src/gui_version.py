from tkinter.constants import BOTTOM, W
import numpy as np
import numpy
from json import JSONEncoder

from NN import *
import json
import tkinter as tk
import tkinter.messagebox as tkmsg

import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename, asksaveasfilename





DEBUG = True

class GUI:
    
    class MainWindowC:
        def __init__(self):
            self.main = tk.Tk()
            self.model = Model()
            self.ModelBiasState = True
            self.NameOfFile = None
            self.DataJsonRaw = {}
            self.modelEvaluationTrain = []
            self.modelEvaluationTest = []
            self.model.learningFactor=0.25
            self.model.iterations= 250
            self.DataFormatOk= False
            self.showFrame = None

            self.topFrame = tk.Frame(self.main)
            self.topFrame.pack()

            self.bottomFrame = tk.Frame(self.main)
            self.bottomFrame.pack(side=BOTTOM)
            print(self.AskForFilePath())
            print(self.AskSaveAsFileName())
            ###
            self.model.addLayer(4)
            self.model.addLayer(3)
            self.model.addLayer(2)
            self.model.addLayer(1)
            self.model.compileModel()

            ###
            

            self.TopLabelFrame = tk.LabelFrame(self.topFrame,text="Compiled Structure of the network",padx=30,pady=12)
            self.TopLabelFrame.grid(column=0,columnspan=300,row=3,rowspan=256,sticky='nswe')
            if len(self.model.structure)==0:
                self.temp = tk.Label(self.TopLabelFrame,text="Compiled Structure is empty")
                self.temp.grid(row=1,column=1)

            self.LabelFrameStructure = []
            xx=0
            row = 2;col=1
            for x,i in enumerate(self.model.structure):
                self.LabelFrameStructure.append(tk.LabelFrame(self.TopLabelFrame,text = "Input Layer" if x==0 else str(x)+". Hidden Layer"))
                tk.Label(self.LabelFrameStructure[-1],text=str(x)).pack()
                tk.Label(self.LabelFrameStructure[-1],text = str(i.matrix.shape[0])).pack()
                self.LabelFrameStructure[-1].grid(row=row,column=col)
                row+=1
                xx=x

            self.LabelFrameStructure.append(tk.LabelFrame(self.TopLabelFrame,text = "Output layer"))
            tk.Label(self.LabelFrameStructure[-1],text=xx+1).pack()
            tk.Label(self.LabelFrameStructure[-1],text=str(self.model.structure[xx].matrix.shape[1])).pack()
            self.LabelFrameStructure[-1].grid(row=row,column=col)
         

            self.main.geometry("1000x800")
            self.create_widgets()
            
            self.main.mainloop()


        def create_widgets(self):
            
            self.labelTitle = tk.Label(self.topFrame,text="NNFS")
            self.labelTitle.config(font=("FreeSans",30))
            self.labelTitle.grid(row=1,column=0,columnspan=1,rowspan=2)
            padx = 3
            temp_col = 1
            row = 1
            self.buttonImportDataFromFile = tk.Button(self.topFrame, text="Import Data From File", command=self.ImportDataFromFile)
            self.buttonImportDataFromFile.grid(row=row,column=temp_col,padx=padx,sticky='nswe')
            temp_col+=1

            self.buttonCreateLayer = tk.Button(self.topFrame, text="Create layer", command=self.CreateNewLayer)
            self.buttonCreateLayer.grid(row=row,column=temp_col,padx=padx,sticky='nswe')
            temp_col+=1

            self.buttonCompileNetwork = tk.Button(self.topFrame, text="Compile Network", command= self.CompileNetwork)
            self.buttonCompileNetwork.grid(row=row,column=temp_col,padx=padx,sticky='nswe')
            temp_col+=1

            self.buttonTrainNetwork = tk.Button(self.topFrame, text="Train Network", command= self.TrainNetwork)
            self.buttonTrainNetwork.grid(row=row,column=temp_col,padx=padx,sticky='nswe')
            temp_col+=1

            self.buttonExportNetworkFromFile = tk.Button(self.topFrame, text="Export Network From File",command=self.ExportNetworkToFile)
            self.buttonExportNetworkFromFile.grid(row=row,column=temp_col,padx=padx,sticky='nswe')
            temp_col=1
            row = 2
            
            self.buttonSettings = tk.Button(self.topFrame, text="Settings", command=self.Settings)
            self.buttonSettings.grid(row=row,column=temp_col,padx=padx,sticky='nswe')
            temp_col+=1

            self.buttonEvaluate = tk.Button(self.topFrame, text="Evaluate",command = self.Evaluate)
            self.buttonEvaluate.grid(row=row,column=temp_col,padx=padx,sticky='nswe')
            temp_col+=1

            self.buttonTrainStatistics = tk.Button(self.topFrame, text="Train Statistics",command=self.TrainStatistics)
            self.buttonTrainStatistics.grid(row=row,column=temp_col,padx=padx,sticky='nswe')
            temp_col+=1

            self.buttonImportNetworkFromFile = tk.Button(self.topFrame, text="Import Network From File",command=self.ImportNetworkFromFile)
            self.buttonImportNetworkFromFile.grid(row=row,column=temp_col+1,padx=padx,sticky='nswe')
            temp_col+=1
            
            del(temp_col)


            self.LabelDEBUG = tk.Label(self.bottomFrame, text="EMPTY")
            self.LabelDEBUG.grid(row=0,column=0,padx=padx)

            self.buttonImportFromFile = tk.Button(self.bottomFrame, text="TE                EST", command=self.DEBUG_helper)
            self.buttonImportFromFile.grid(row=1,column=0,padx=padx)


        def ImportNetworkFromFile(self):
            data=json.loads(self.ExportNetworkToFile())
            self.model.Import(data['model'])
            self.modelEvaluationTest = data['evaluate_test']
            self.modelEvaluationTrain = data['evaluate_train']

        def ExportNetworkToFile(self):
            temp = {
                'model': self.model.Export(),
                'evaluate_train' : self.modelEvaluationTrain,
                'evaluate_test' : self.modelEvaluationTest
            }
            print('####\n'+str(temp))
            return json.dumps(temp, cls=NumpyArrayEncoder)
             

        def Evaluate(self):
            if not len(self.model.structure)>0:
                tkmsg.showwarning("Network not compiled","Please compile network to be able to test it on a data!")
                return
            
            self.EvaluateTopLevel = tk.Toplevel()

            self.EvaluateLabelMainLabel = tk.Label(self.EvaluateTopLevel,text="Input data to the network.\n\tWrite Input data separetet by comma.\n")
            self.EvaluateLabelMainLabel.grid(row=1,column=0,columnspan=2)

            self.EvaluateEntryInput = tk.Entry(self.EvaluateTopLevel,width=30)
            self.EvaluateEntryInput.grid(row=2,column=0,columnspan=2)

            self.EvaluateLabelOutput = tk.Label(self.EvaluateTopLevel,text = "\nYour network output: \n")
            self.EvaluateLabelOutput.grid(row=3,column=0,columnspan=2)
            
            self.EvaluateButtonCancel = tk.Button(self.EvaluateTopLevel,text="Close Window",command=self.EvaluateTopLevel.destroy)
            self.EvaluateButtonCancel.grid(row=30,column=0 )
                        #TODO
            self.EvaluateButtonSubmit = tk.Button(self.EvaluateTopLevel,text="Submit",command=self.EvaluateReadAndForward)
            self.EvaluateButtonSubmit.grid(row=30,column=1 )

        def EvaluateReadAndForward(self):
            inputData = self.EvaluateEntryInput.get()
            try:
                inputData = inputData.split(',')
                for i in range(len(inputData)):
                    inputData[i] = float(inputData[i])
                # print('float')
                if not self.model.structure[0].matrix.shape[0] == len(inputData):
                    raise Exception()
                # print('dims')
                output = self.model.forward(inputData)
                #print('forward')
                self.EvaluateLabelOutput['text']="\nYour network output: \n"+'\n'.join([str(i) for i in output])+'\n'
            except Exception as ex:
                print(ex)
                tkmsg.showwarning("Input data wrong format","Please correct input data to match dims of network!")
                return

        def ShowNetworkStructure(self ):
            if self.showFrame == None:
                self.showFrame = tk.Frame(self.topFrame)
        def DEBUG_helper(self):
            #self.ShowNetworkStructure()
            value = "self.model.layers"+ str(self.model.layers)+"\n"
            value += "self.model.structure"+str(self.model.structure)+"\n"
            value += "\n Layers:\n"
            for i in self.model.structure:
                value += str(i.matrix.shape)+"\n"

            value += "Dataset: "+ str(self.model.dataset)+"\n"
            self.LabelDEBUG["text"] = value


        #####
        def TrainStatistics(self):

            if False and DEBUG :
                self.modelEvaluationTrain = np.linspace(0,1,100)
                self.modelEvaluationTest = np.linspace(-1,3,100)
            

            
            if self.modelEvaluationTest is None or len(self.modelEvaluationTest) == 0 or self.modelEvaluationTrain is  None or len(self.modelEvaluationTrain) == 0:
                tkmsg.showinfo("Cannot calculate statistics","Model is not trained or initailized, so it's imposible to calculate values and make charts!")
                return
            self.TrainStatisticsTopLevel = tk.Toplevel()
            
            temp_row=1
            self.TrainStatisticsLabelMainText = tk.Label(self.TrainStatisticsTopLevel,text = "Basic statistics form network training:")
            self.TrainStatisticsLabelMainText.grid(column=1,row=temp_row)
            temp_row+=1

            self.TrainStatisticsLabelTrain = tk.Label(self.TrainStatisticsTopLevel,text = "Minimum Train Error: "+str(np.min(self.modelEvaluationTrain))+\
                "\nAverage Train Error: "+str(np.average(self.modelEvaluationTrain))+"\nMaximum Train Error: "+str(np.max(self.modelEvaluationTrain)))
            self.TrainStatisticsLabelTrain.grid(column=1,row=temp_row)
            temp_row+=1

            self.TrainStatisticsLabelTest = tk.Label(self.TrainStatisticsTopLevel,text = "Minimum Test Error: "+str(np.min(self.modelEvaluationTest))+\
                "\nAverage Test Error: "+str(np.average(self.modelEvaluationTest))+"\nMaximum Test Error: "+str(np.max(self.modelEvaluationTest)))
            self.TrainStatisticsLabelTest.grid(column=1,row=temp_row)
            temp_row+=1

            

            self.TrainStatisticsButtonErrorPlotShow = tk.Button(self.TrainStatisticsTopLevel,text = "Interactive error plot",command=self.ErrorPlotShow)
            self.TrainStatisticsButtonErrorPlotShow.grid(column=1,row=temp_row)
            temp_row+=1

            del(temp_row)


           

            

        def ErrorPlotShow(self):
            plt.plot(self.modelEvaluationTrain)
            plt.plot(self.modelEvaluationTest)
            plt.show()



        def ImportDataFromFile(self):
            self.DataFormatOk= False
            nameoffile = self.AskForFilePath()
            if nameoffile == ():
                #tkmsg.showwarning("File not selected", "File was not selected. Please select file to import data!")
                return
            if nameoffile.split('.')[-1] == 'json':
                
                try:
                    with open(nameoffile ) as file:
                        self.DataJsonRaw=json.load(file)
                except Exception as ex:
                    print(ex)
                    tkmsg.showerror("Wrong file format", "File format corrupteed! Cannot parse data!")
                    return
            else:
                tkmsg.showerror("Wrong file extention", "Unrecognized file extention!")
                return

            if not self.ImportDataFromFilecheckDataFormatIsOk():
                tkmsg.showerror("Data wrong format!", "Data read form file is in wrong format!")
                return 
            else:
                self.DataChecked = dict(self.DataJsonRaw)
                self.model.dataset = dict(self.DataChecked)
                
        def ImportDataFromFilecheckDataFormatIsOk(self):
            if not isinstance(self.DataJsonRaw,dict) or (not 'train' in self.DataJsonRaw.keys()) or (not 'test' in self.DataJsonRaw.keys())\
                or (not "egzo" in self.DataJsonRaw['train'].keys()) or (not "endo" in self.DataJsonRaw['train'].keys()) or (not "egzo" in self.DataJsonRaw['test'].keys())\
                or (not "endo" in self.DataJsonRaw['test'].keys()) or len(self.DataJsonRaw['train']['egzo'])!= len(self.DataJsonRaw['train']['endo'])\
                or (len(self.DataJsonRaw['test']['egzo'])!= len(self.DataJsonRaw['test']['endo'])):
                return False
            return True
                

        def AskForFilePath(self):
            return askopenfilename() 

        def AskSaveAsFileName(self,filetypes=[("json file", '*.json'),("None",'*')]):
            return asksaveasfilename( filetypes=filetypes) 

        def CreateNewLayer(self):
            self.CreateNewLayerTopLevel = tk.Toplevel()
            # self.CreateNewLayerTopLevel.geometry("400x150")



            self.CreateNewLayerLabelNumberOfNeurons = tk.Label(self.CreateNewLayerTopLevel,text="Number of neurons: ")
            self.CreateNewLayerLabelNumberOfNeurons.config(font=("FreeSans",16))
            self.CreateNewLayerLabelNumberOfNeurons.grid(row=1,column=1, columnspan=2)


            #entry

            self.CreateNewLayerInputNumberOfNeurons = tk.Entry(self.CreateNewLayerTopLevel,width=20)
            self.CreateNewLayerInputNumberOfNeurons.grid(row=2,column=1, columnspan=2)


            #radiobuttons
            self.CreateNewLayerActivationFuncitonVariable=tk.IntVar()
            self.CreateNewLayerActivationFuncitonVariable.set(1)

            self.CreateNewLayerRadioButtonUnipolar = tk.Radiobutton(self.CreateNewLayerTopLevel, text="Sigmoid (Unipolar)",variable = self.CreateNewLayerActivationFuncitonVariable, value=1)
            self.CreateNewLayerRadioButtonUnipolar.grid(row=3,column=1, columnspan=2, sticky="W")


            self.CreateNewLayerRadioButtonBipolar = tk.Radiobutton(self.CreateNewLayerTopLevel, text="ArcTG (Bipolar)",variable = self.CreateNewLayerActivationFuncitonVariable, value=2)
            self.CreateNewLayerRadioButtonBipolar.grid(row=4,column=1, columnspan=2, sticky="W")



            self.CreateNewLayerButtonSubmit = tk.Button(self.CreateNewLayerTopLevel,text="Submit",command=self.CreateNewLayerReadFieldsAndExit)
            self.CreateNewLayerButtonSubmit.grid(row=5,column=2 )
            
            self.CreateNewLayerButtonCancel = tk.Button(self.CreateNewLayerTopLevel,text="Cancel",command=self.CreateNewLayerTopLevel.destroy)
            self.CreateNewLayerButtonCancel.grid(row=5,column=1 )



        def CreateNewLayerReadFieldsAndExit(self):
            #validate input, raise messagebox to warn user to type correct input
            temp_AF = self.CreateNewLayerActivationFuncitonVariable.get() 
            print("activaton: ",self.CreateNewLayerActivationFuncitonVariable.get())
            print("Number of neurons",self.CreateNewLayerInputNumberOfNeurons.get())
            try:
                self.CreateNewLayerNumberOfNeurons = int(self.CreateNewLayerInputNumberOfNeurons.get())
                assert self.CreateNewLayerNumberOfNeurons >=1
                assert self.CreateNewLayerNumberOfNeurons <=250
                
                if temp_AF == 1:
                    temp_AF = ActivationFunctions.unipolar
                else:
                    temp_AF = ActivationFunctions.bipolar
                self.model.addLayer(self.CreateNewLayerNumberOfNeurons,temp_AF)
                del(temp_AF)
                self.CreateNewLayerTopLevel.destroy()
            except Exception :
                tkmsg.showerror("Invalid Input!",'Please make sure, number of neurons in this layer is correct!')
                return 
        
        def TrainNetworkMonitor(self,iteration:int = None,wholeError_train:float=0.0,wholeError_test:float=0.0,correction = None):
            self.modelEvaluationTrain.append(float(wholeError_train))
            self.modelEvaluationTest.append(float(wholeError_test))


        def TrainNetwork(self):
            ok=True


            if not self.DataFormatOk:
                for y in ['train','test']:
                    shape =  self.model.structure[0].matrix.shape[0]
                    for i in self.model.dataset[y]['egzo']:
                        if len(i)!=shape:
                            ok=False
                            break
                    if not ok:
                        break


            if not self.DataFormatOk and ok:
                for y in ['train','test']:
                    shape =  self.model.structure[-1].matrix.shape[-1]
                    for i in self.model.dataset[y]['endo']:
                        if len(i)!=shape:
                            ok=False
                            break
                    if not ok:
                        break


            if not ok:
                tkmsg.showerror("Data and structure", "Dimentions of data and network don't match!")
                return
            else:
                self.DataFormatOk = True
            self.modelEvaluationTrain = []
            self.modelEvaluationTest = []
            if len(self.model.structure) == 0 or self.model.dataset ==None or self.model.dataset=={} or (not "train" in self.model.dataset.keys()) or (not "train" in self.model.dataset.keys()):
                tkmsg.showerror("Uninitialized Network", "Please make sure the dataset is imported and network is compiled!")
                return 
            self.model.setExternalMonitor(self.TrainNetworkMonitor)
            # TODO add including monitor
            self.model.train()
            tkmsg.showinfo("Model Trained!","Model is trained!")
            


        def CompileNetwork(self):
            if not len(self.model.layers)>=2:
                tkmsg.showerror("Too few layers!", "Network requiers at least 2 layers, input and output. Please add them!")
                return
            self.model.compileModel()
            if DEBUG:
                tkmsg.showinfo("compiled!","Model was compiled!")
        def Settings(self):
            self.SettingsTopLevel = tk.Toplevel()
            
            self.SettingsLabelMainText = tk.Label(self.SettingsTopLevel,text = "Settings: ")
            self.SettingsLabelMainText.grid(row=1,column=1,sticky="W", columnspan=2)


            self.SettingsCheckBoxBiasVar = tk.IntVar()
            self.SettingsCheckBoxBiasVar.set(1 if self.ModelBiasState else 0)
            
            self.SettingsCheckBoxBias = tk.Checkbutton(self.SettingsTopLevel, text='use Bias',variable=self.SettingsCheckBoxBiasVar, onvalue=1, offvalue=0)
            self.SettingsCheckBoxBias.grid(row=2,column=1,sticky="W")

            #learning factor
            self.SettingsLearningFactorLabel = tk.Label(self.SettingsTopLevel,text="Learning Factor")
            self.SettingsLearningFactorLabel.grid(row=3,column=1)

            self.SettingsLearningFactorEntry = tk.Entry(self.SettingsTopLevel)
            self.SettingsLearningFactorEntry.insert(0, str(self.model.learningFactor))
            self.SettingsLearningFactorEntry.grid(row=4,column=1)
            
            #iterations
            self.SettingsIterationsLabel = tk.Label(self.SettingsTopLevel,text="Iterations: ")
            self.SettingsIterationsLabel.grid(row=5,column=1)

            self.SettingsIterationsEntry = tk.Entry(self.SettingsTopLevel, width = 20)
            self.SettingsIterationsEntry.insert( 0, str(self.model.iterations))
            self.SettingsIterationsEntry.grid(row=6,column=1)



            
            self.SettingsButtonCancel = tk.Button(self.SettingsTopLevel,text="Cancel",command=self.SettingsTopLevel.destroy)
            self.SettingsButtonCancel.grid(row=30,column=1 )
                        #TODO
            self.SettingsButtonSubmit = tk.Button(self.SettingsTopLevel,text="Submit",command=self.SettingsReadFieldsAndExit)
            self.SettingsButtonSubmit.grid(row=30,column=2 )



        def SettingsReadFieldsAndExit(self):
            #parsing data
            iterations = self.SettingsIterationsEntry.get()
            try:
                iterations = int(iterations)
                if iterations <1:
                    raise Exception()
            except Exception as ex:
                tkmsg.showerror("Wrong number of iterations","Wrong number of iterations")
                return
            learningRate = self.SettingsLearningFactorEntry.get()
            try:
                learningRate = float(learningRate)
                if learningRate <=0:
                    raise Exception()
            except Exception as ex:
                tkmsg.showerror("Wrong learning factor","Wrong learning factor! Please pass correct number!")
                return


            if self.SettingsCheckBoxBiasVar.get()==1:
                self.ModelBiasState = True
            else: 
                self.ModelBiasState = False
            self.model.iterations = iterations
            self.model.learningFactor = learningRate

            self.model.biasState = self.ModelBiasState
            self.SettingsTopLevel.destroy()

            


            
    
    def __init__(self):
        self.mainWindowO = self.MainWindowC()
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        print(type(obj))
        if isinstance(obj, numpy.ndarray) or isinstance(obj, numpy.array):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

if __name__ == "__main__":
    GUI()

                    
'''                   

    
    performance = []
    model = Model()

    model.addLayer(4,ActivationFunctions.bipolar)
    model.addLayer(4,ActivationFunctions.bipolar)
    model.addLayer(1,ActivationFunctions.bipolar)


    model.compileModel()

    train_dataset= {
        'egzo': np.array([[1,1,0,0],[1,0,0,0],[1,1,1,1]]), 
        'endo':np.array([ [1],      [0],      [0.5]])
    }


    model.upload_train_dataset(train_dataset)
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



'''
