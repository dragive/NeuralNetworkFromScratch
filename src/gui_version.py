from tkinter.constants import BOTTOM
import numpy as np

from NN import *
import json
import tkinter as tk
import tkinter.messagebox as tkmsg

import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename





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

            self.topFrame = tk.Frame(self.main)
            self.topFrame.pack()

            self.bottomFrame = tk.Frame(self.main)
            self.bottomFrame.pack(side=BOTTOM)
            
            self.main.geometry("1000x800")
            self.create_widgets()
            
            self.main.mainloop()

            
        def create_widgets(self):
            
            self.labelTitle = tk.Label(self.topFrame,text="NNFS")
            self.labelTitle.config(font=("FreeSans",30))
            self.labelTitle.grid(row=0,column=0)
            padx = 3
            temp_col = 1

            self.buttonImportDataFromFile = tk.Button(self.topFrame, text="Import Data From File", command=self.ImportDataFromFile)
            self.buttonImportDataFromFile.grid(row=0,column=temp_col,padx=padx)
            temp_col+=1

            self.buttonCreateLayer = tk.Button(self.topFrame, text="Create layer", command=self.CreateNewLayer)
            self.buttonCreateLayer.grid(row=0,column=temp_col,padx=padx)
            temp_col+=1

            self.buttonCompileNetwork = tk.Button(self.topFrame, text="Compile Network", command= self.CompileNetwork)
            self.buttonCompileNetwork.grid(row=0,column=temp_col,padx=padx)
            temp_col+=1

            self.buttonTrainNetwork = tk.Button(self.topFrame, text="Train Network", command= self.TrainNetwork)
            self.buttonTrainNetwork.grid(row=0,column=temp_col,padx=padx)
            temp_col+=1
            
            self.buttonSettings = tk.Button(self.topFrame, text="Settings", command=self.Settings)
            self.buttonSettings.grid(row=0,column=temp_col,padx=padx)
            temp_col+=1

            self.buttonEvaluate = tk.Button(self.topFrame, text="Evaluate")
            self.buttonEvaluate.grid(row=0,column=temp_col,padx=padx)
            temp_col+=1

            self.buttonTrainStatistics = tk.Button(self.topFrame, text="Train Statistics",command=self.TrainStatistics)
            self.buttonTrainStatistics.grid(row=0,column=temp_col,padx=padx)
            temp_col+=1
            
            del(temp_col)

            self.LabelDEBUG = tk.Label(self.bottomFrame, text="EMPTY")
            self.LabelDEBUG.grid(row=0,column=0,padx=padx)

            self.buttonImportFromFile = tk.Button(self.bottomFrame, text="TE                EST", command=self.DEBUG_helper)
            self.buttonImportFromFile.grid(row=1,column=0,padx=padx)
            
        def DEBUG_helper(self):
            value = "self.model.layers"+ str(self.model.layers)+"\n"
            value += "self.model.structure"+str(self.model.structure)+"\n"
            value += "\n Layers:\n"
            for i in self.model.structure:
                value += str(i.matrix.shape)+"\n"

            value += "Dataset: "+ str(self.model.dataset)+"\n"
            self.LabelDEBUG["text"] = value


        #####
        def TrainStatistics(self):
            self.TrainStatisticsTopLevel = tk.Toplevel()
            if False and  DEBUG and( not len(self.modelEvaluationTrain)==0 or not len(self.modelEvaluationTest)==0):
                self.modelEvaluationTrain = np.linspace(0,1,100)
                self.modelEvaluationTest = np.linspace(-1,3,100)
            
            if self.modelEvaluationTest is None or len(self.modelEvaluationTest) == 0 or self.modelEvaluationTrain is  None or len(self.modelEvaluationTrain) == 0:
                tkmsg.showinfo("Cannot calculate statistics","Model is not trained or initailized, so it's imposible to calculate values and make charts!")
                return
            print("###")
            #self.TrainStatisticsLabel
            temp_row=1
            self.TrainStatisticsLabelMainText = tk.Label(self.TrainStatisticsTopLevel,text = "Basic statistics form network training:")
            self.TrainStatisticsLabelMainText.grid(column=1,row=temp_row)
            temp_row+=1

            self.TrainStatisticsLabelMinTrainError = tk.Label(self.TrainStatisticsTopLevel,text = "Minimum Train Error: "+str(np.min(self.modelEvaluationTrain)))
            self.TrainStatisticsLabelMinTrainError.grid(column=1,row=temp_row)
            temp_row+=1

            self.TrainStatisticsLabelAverageTrainError = tk.Label(self.TrainStatisticsTopLevel,text = "Average Train Error: "+str(np.average(self.modelEvaluationTrain)))
            self.TrainStatisticsLabelAverageTrainError.grid(column=1,row=temp_row)
            temp_row+=1

            self.TrainStatisticsLabelMaxTrainError = tk.Label(self.TrainStatisticsTopLevel,text = "Maximum Train Error: "+str(np.max(self.modelEvaluationTrain)))
            self.TrainStatisticsLabelMaxTrainError.grid(column=1,row=temp_row)
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
            nameoffile = self.ImportDataFromFileAskForFilePath()
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
                

        def ImportDataFromFileAskForFilePath(self):
            return askopenfilename() 

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
            self.modelEvaluationTrain.append(wholeError_train)
            self.modelEvaluationTest.append(wholeError_test)


        def TrainNetwork(self):
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
            
            self.SettingsButtonCancel = tk.Button(self.SettingsTopLevel,text="Cancel",command=self.SettingsTopLevel.destroy)
            self.SettingsButtonCancel.grid(row=30,column=1 )
                        #TODO
            self.SettingsButtonSubmit = tk.Button(self.SettingsTopLevel,text="Submit",command=self.SettingsReadFieldsAndExit)
            self.SettingsButtonSubmit.grid(row=30,column=2 )



        def SettingsReadFieldsAndExit(self):
            if self.SettingsCheckBoxBiasVar.get()==1:
                self.ModelBiasState = True
            else: 
                self.ModelBiasState = False
            
            self.model.biasState = self.ModelBiasState
            self.SettingsTopLevel.destroy()

            


            
    
    def __init__(self):
        self.mainWindowO = self.MainWindowC()

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