#!/usr/bin/python3
from os import name
from tkinter.constants import BOTTOM

import numpy as np
import numpy
from json import JSONEncoder
import copy
from numpy.lib.shape_base import column_stack

from numpy.random import default_rng
from NN_model import *
import json
import tkinter as tk
import tkinter.messagebox as tkmsg

import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename, asksaveasfilename

import re
import os


DEBUG = False

class GUI:
    
    class MainWindowC:
        def __init__(self):
            self.main = tk.Tk()

            try:
                self.main.tk.call('wm', 'iconphoto', self.main._w, tk.PhotoImage(file=os.path.realpath('ico.png')))
            except Exception as ex:
                pass

            self.main.title("Neural Networks From Scratch")
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
            self.LayerMonitorWidgetStorage={}
            self.topFrame = tk.Frame(self.main)
            self.topFrame.pack()
            self.LabelFrameStructure = []
            self.bottomFrame = tk.Frame(self.main)
            self.bottomFrame.pack(side=BOTTOM)
            self.DataManagerStorage = {}

            self.main.protocol("WM_DELETE_WINDOW", self.on_closing)
            ###
            self.GenerateCompiledStructureWidget()
            self.GenerateNotCompiledStructureWidget()
            
            self.Update()
            ##

            self.main.geometry("1000x800")
            self.create_widgets()
            
            self.main.mainloop()

        def on_closing(self):
            if tkmsg.askokcancel("Quit", "Do you want to quit?"):
                self.main.destroy()

            
        def GenerateNotCompiledStructureWidget(self):
            self.TopLabelFrameNotCompiled = tk.LabelFrame(self.topFrame,text="Not-Compiled Structure of the Network")
            self.TopLabelFrameNotCompiled.grid(column=0,columnspan=300,row=3,rowspan=1,sticky='nswe',pady=10,padx=5)
            
            self.TopLabelFrameNotCompiledLabelStructure = tk.Label(self.TopLabelFrameNotCompiled,text = "[Input Layer] [Output Layer]")
            self.TopLabelFrameNotCompiledLabelStructure.grid(row=10,column=10,columnspan=200,padx=5,pady=5)


            self.TopLabelFrameNotCompiledButtonClear = tk.Button(self.TopLabelFrameNotCompiled,text = "Clear Structure", command=self.ModelClearLayers)
            self.TopLabelFrameNotCompiledButtonClear.grid(row=5,column=10,sticky='nsw',pady=5,padx=5)

            self.UpdateNotCompiledStructureWidget()
        def Update(self):
            self.UpdateNotCompiledStructureWidget()
            self.UpdateCompiledStructureWidget()
        def ModelClearLayers(self):
            self.model.layers = []
            self.Update()
        def UpdateNotCompiledStructureWidget(self):
            ls = self.model.layers
            ret = "[Input Layer]  "
            
            ret += ' -> '.join(['('+str(y[0])+', '+ActivationFunctions.Export(y[1])+')' for y in ls])
                
            ret += "  [Output Layer]"
            self.TopLabelFrameNotCompiledLabelStructure['text'] = ret

        def GenerateCompiledStructureWidget(self):
            
            self.TopLabelFrameCompiled = tk.LabelFrame(self.topFrame,text="Compiled Structure of the Network",padx=30,pady=12)
            self.TopLabelFrameCompiled.grid(column=0,columnspan=300,row=4,rowspan=256,sticky='nswe',pady=10,padx=5)
            self.UpdateCompiledStructureWidgetEmpty()

            
            
            self.UpdateCompiledStructureWidget()


        def UpdateCompiledStructureWidgetEmpty(self):
            if len(self.model.structure)==0 :#or len(self.LabelFrameStructure)==0:
                # print('EMPTY?????')
                self.templabel1 = tk.Label(self.TopLabelFrameCompiled,text="Compiled Structure is empty")
                self.templabel1.grid(row=1,column=1)
            else:
                # print('NOOOOOOOT EMPTY?????')
                self.templabel1.destroy()

        def LayerMonitorWidgetSetupOnClose(self,number):
            self.LayerMonitorWidgetStorage[number]['toplevel'].destroy()
            del(self.LayerMonitorWidgetStorage[number])
            
        def LayerMonitorWidgetSetup(self,number):
            storage = self.LayerMonitorWidgetStorage
            if not number in storage.keys():
                storage [number] = {}
            else:
                storage[number]['toplevel'].lift()
                return
            st =storage[number]
            st['toplevel'] = tk.Toplevel()
            st['toplevel'].title('Layer Monitor')
            st['toplevel'].protocol("WM_DELETE_WINDOW", lambda number=number:self.LayerMonitorWidgetSetupOnClose(number))
        
            st['labeltitle'] = tk.Label(st['toplevel'],text="Hidden layer "+str(number+1)+'\n' if number +1 !=len( self.model.structure) else 'Output Layer')
            st['labeltitle'].config(font=("FreeSans",15))
            st['labeltitle'].grid(row=1,column = 1)

            st['structure'] = tk.Label(st['toplevel'],text = '\n')
            st['structure'].grid(row=2,column = 1)
            self.LayerMonitorWidgetSetupUpdate()
            st['toplevel'].lift()
        
        def LayerMonitorWidgetSetupUpdate(self):
            storage = self.LayerMonitorWidgetStorage
            for k in storage.keys():
                number = k
                if self.model.structure[number].bias.ndim >1:
                    self.model.structure[number].bias = self.model.structure[number].bias[0]
                if len( self.model.structure[number].dot) == 1000:
                    self.model.structure[number].dot = np.zeros(self.model.structure[number].shape[1])
                if  self.model.structure[number].output is None:
                    self.model.structure[number].output = np.zeros(self.model.structure[number].shape[1])
                text = ''
                if self.model.biasState:
                    text=  '\n '.join(
                    [' LC = '+' + '.join(
                                [f'({y:7.4}*x{z+1})' for z,y in enumerate(i)]
                                )
                                    + f' + ( {b :7.4} ) '# = {d :7.4} -> f(a) =  {z :7.4}' 
                                        for i,b,d,z in zip(self.model.structure[number].matrix.tolist(),self.model.structure[number].bias.tolist(),\
                                            self.model.structure[number].dot.tolist(),self.model.structure[number].output.tolist())]
                    )
                else:
                    text =  '\n '.join(
                    [''+' + '.join(
                                [f'({y:7.4}*x{z+1})' for z,y in enumerate(i)]
                                )
                                    + f''# = {d :7.4} -> f(a) =  {z :7.4}' 
                                        for i,b,d,z in zip(self.model.structure[number].matrix.tolist(),self.model.structure[number].bias.tolist(),\
                                            self.model.structure[number].dot.tolist(),self.model.structure[number].output.tolist())]
                    )
                try:
                    self.model.structure[0].input = [float(iii) for iii in self.model.structure[0].input]
                except Exception as ex:
                    pass
                text +='\n\n\n'
                if not (self.model.structure[number].input is None or len(self.model.structure[number].input)==1000):
                    text+='Input: '+', '.join([f'{foo :7.4}' for foo in self.model.structure[number].input])+'\n'
                
                if not (self.model.structure[number].dot is None or len(self.model.structure[number].dot)==1000):
                    text+='Linear combination: '+', '.join([f'{foo :7.4}' for foo in self.model.structure[number].dot])+'\n'
                
                if not (self.model.structure[number].output is None or len(self.model.structure[number].output)==1000):
                    text+='Output: '+', '.join([f'{foo :7.4}' for foo in self.model.structure[number].output])+'\n'
                storage[k]['structure']['text']=text
                
            
            

        def UpdateCompiledStructureWidget(self):
            
            self.ClearCompiledStructureWigdet()
            # self.UpdateCompiledStructureWidgetEmpty()
            if len(self.model.structure)==0:
                return
            xx=0
            row = 2;col=1

            self.LabelFrameStructure.append(tk.LabelFrame(self.TopLabelFrameCompiled,text = "Input layer"))
            tk.Label(self.LabelFrameStructure[-1],text=f"Number of neurons: {self.model.structure[xx].matrix.shape[0]}").grid(row=1,column=1,sticky = "wn")
            
            
            self.LabelFrameStructure[-1].grid(row=row,column=col,sticky= 'new')
            row+=1
            

            xx=0

            for i in self.model.structure:
                
                lamb = lambda xx=xx:self.LayerMonitorWidgetSetup(xx)
                self.LabelFrameStructure.append(tk.LabelFrame(self.TopLabelFrameCompiled,text = "Output Layer" if xx==len(self.model.structure)-1 else str(xx+1)+". Hidden Layer"))
                tk.Label(self.LabelFrameStructure[-1],text=f"Number of neurons: {self.model.structure[xx].matrix.shape[1]}").grid(row=1,column=1,sticky = "wn")
                tk.Label(self.LabelFrameStructure[-1],text=f"Activation Funciton: {ActivationFunctions.Export( self.model.structure[xx].function)}").grid(row=2,column=1,sticky = "wn")
                tk.Button(self.LabelFrameStructure[-1],text="See details",command = lamb).grid(row=3,column=1,sticky='news')
                self.LabelFrameStructure[-1].grid(row=row,column=col)
                row+=1
                xx=xx+1
            self.UpdateCompiledStructureWidgetEmpty()
            ##
            # for x,i in enumerate(self.model.structure):
            #     self.LabelFrameStructure.append(tk.LabelFrame(self.TopLabelFrameCompiled,text = "Input Layer" if x==0 else str(x)+". Hidden Layer"))
            #     tk.Label(self.LabelFrameStructure[-1],text=str(x)).pack()
            #     tk.Label(self.LabelFrameStructure[-1],text = str(i.matrix.shape[0])).pack()
            #     self.LabelFrameStructure[-1].grid(row=row,column=col)
            #     row+=1
            #     xx=x

            # self.LabelFrameStructure.append(tk.LabelFrame(self.TopLabelFrameCompiled,text = "Output layer"))
            # tk.Label(self.LabelFrameStructure[-1],text=xx+1).pack()
            # tk.Label(self.LabelFrameStructure[-1],text=str(self.model.structure[xx].matrix.shape[1])).pack()
            # self.LabelFrameStructure[-1].grid(row=row,column=col)
            
        def ClearCompiledStructureWigdet(self):
            for i in self.LabelFrameStructure:
                i.destroy()
            self.LabelFrameStructure=[]
        def create_widgets(self):
            
            self.labelTitle = tk.Label(self.topFrame,text="NNFS")
            # self.labelTitle.config(font=("FreeSans",30))
            # self.labelTitle.grid(row=1,column=0,columnspan=1,rowspan=2)
            padx = 3
            temp_col = 1
            row = 1

            self.buttonManageDataFile = tk.Button(self.topFrame, text="Manage Data",command=self.DataManager)
            self.buttonManageDataFile.grid(row=row,column=temp_col,padx=padx,sticky='nswe')
            temp_col+=1

            # self.buttonImportDataFromFile = tk.Button(self.topFrame, text="Import Data From File", command=self.ImportDataFromFile)
            # self.buttonImportDataFromFile.grid(row=row,column=temp_col,padx=padx,sticky='nswe')
            # temp_col+=1

            self.buttonCreateLayer = tk.Button(self.topFrame, text="Create layer", command=self.CreateNewLayer)
            self.buttonCreateLayer.grid(row=row,column=temp_col,padx=padx,sticky='nswe')
            temp_col+=1

            self.buttonCompileNetwork = tk.Button(self.topFrame, text="Compile Network", command= self.CompileNetwork)
            self.buttonCompileNetwork.grid(row=row,column=temp_col,padx=padx,sticky='nswe')
            temp_col+=1

            self.buttonTrainNetwork = tk.Button(self.topFrame, text="Train Network", command= self.TrainNetwork)
            self.buttonTrainNetwork.grid(row=row,column=temp_col,padx=padx,sticky='nswe',columnspan=1)
            temp_col+=1

            self.buttonExportNetworkFromFile = tk.Button(self.topFrame, text="Export Network To File",command=self.ExportNetworkToFile)
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

            # self.buttonStep = tk.Button(self.topFrame, text="One Epoch",command=self.Step)
            # self.buttonStep.grid(row=row,column=temp_col,padx=padx,sticky='nswe')
            # temp_col+=1

            self.buttonOneStep = tk.Button(self.topFrame, text="One Iteration",command=self.OneStep)
            self.buttonOneStep.grid(row=row,column=temp_col,padx=padx,sticky='nswe')
            temp_col+=1



            self.buttonImportNetworkFromFile = tk.Button(self.topFrame, text="Import Network From File",command=self.ImportNetworkFromFile)
            self.buttonImportNetworkFromFile.grid(row=row,column=temp_col,padx=padx,sticky='nswe')
            temp_col+=1
            
            del(temp_col)

            if DEBUG:
                self.LabelDEBUG = tk.Label(self.bottomFrame, text="EMPTY")
                self.LabelDEBUG.grid(row=0,column=0,padx=padx)

                self.buttonImportFromFile = tk.Button(self.bottomFrame, text="TE                EST", command=self.DEBUG_helper)
                self.buttonImportFromFile.grid(row=1,column=0,padx=padx)
        def DataManagerImport(self):
            self.ImportDataFromFile()
            self.DataManagerTopLevel.destroy()
            if self.model.dataset is None or self.model.dataset == {}:
                pass
            else:
                tkmsg.showinfo('Data Imported','Data imported correctly')
        def DataManager(self):
            self.DataManagerTopLevel = tk.Toplevel()
            self.DataManagerTopLevel.title('Data Manager')
            self.DataManagerImportData = tk.Button(self.DataManagerTopLevel, text="Import From File",command = self.DataManagerImport)
            self.DataManagerImportData.grid(row=1000,column=1)
            self.DataManagerImportData = tk.Button(self.DataManagerTopLevel, text="Add Data Manualy",command = self.DataManagerAdd)
            self.DataManagerImportData.grid(row=1000,column=2)
            

            
            
            if self.model.dataset is None or self.model.dataset == {}:
                tk.Label(self.DataManagerTopLevel, text="There's no data").grid(row=2,column=1,columnspan=3,pady=30,padx=30)
                return
            tk.Label(self.DataManagerTopLevel, text="Check \"Train\" to include data to train dataset else to test dataset.").grid(row=2,column=1,columnspan=3,pady=20)
            tk.Button(self.DataManagerTopLevel,text="Submit",command=self.DataManagerReadAndExit).grid(row=1000,column=3)
            # print(self.model.dataset)
            self.DataManagerEditFrame=None
            self.DataManagerEditUpdate()
        def DataManagerAdd(self):
            
            index = 0
            self.DataManagerAddTopLevel = tk.Toplevel()
            self.DataManagerAddTopLevel.title('Add Data')
            
            self.DataManagerAddEntry=tk.Entry(self.DataManagerAddTopLevel,width = 40)
            self.DataManagerAddEntry.grid(row=1,column=0,columnspan = 2)
            value =' -> '
            self.DataManagerAddEntry.insert(0,value)

            self.DataManagerEditSubmitBTN = tk.Button(self.DataManagerAddTopLevel,text='Submit',command=self.DataManagerAddSubmit)
            self.DataManagerEditSubmitBTN.grid(row=2,column=0)
        def DataManagerAddSubmit(self):
            value = self.DataManagerAddEntry.get()
            if value == '':
                
                # self.DataManagerAddTopLevel[index].destroy()
                return 
            try:
                egzo ,endo = value.split('->')
                egzo = [float(i) for i in egzo.split(',')]
                endo = [float(i) for i in endo.split(',')]
                self.model.dataset['train']['egzo'].append(egzo)
                self.model.dataset['train']['endo'].append(endo)

            except Exception as ex:
                print(ex)
                tkmsg.showwarning("Data Edit Error","Error occured while editing dataset!")
                return
            # self.DataManagerReadAndExit(destroy=False)
            self.DataManagerEditUpdate()
            # self.DataManagerEditTopLevel[].destroy()
        def DataManagerEditUpdate(self):
            if self.DataManagerEditFrame is None:
                self.DataManagerEditFrame = tk.Frame(self.DataManagerTopLevel)
                self.DataManagerEditFrame.grid(row=3,column=1,columnspan=3)
            else:
                self.DataManagerEditFrame.destroy()
                self.DataManagerEditFrame = tk.Frame(self.DataManagerTopLevel)
                self.DataManagerEditFrame.grid(row=3,column=1,columnspan=3)
            self.DataManagerStorage = {}
            
            index = 0
            for part in ('train','test'):
                for i in range(len(self.model.dataset[part]['egzo'])):
                    self.DataManagerStorage[index]={}

                    self.DataManagerStorage[index]['var'] = tk.StringVar()
                    self.DataManagerStorage[index]['var'].set(part)
                    self.DataManagerStorage[index]['data']={}
                    self.DataManagerStorage[index]['data']['egzo'] = self.model.dataset[part]['egzo'][i]
                    self.DataManagerStorage[index]['data']['endo'] = self.model.dataset[part]['endo'][i]

                    self.DataManagerStorage[index]['checkbox'] = tk.Checkbutton(self.DataManagerEditFrame ,text=' Train ',variable=self.DataManagerStorage[index]['var'], onvalue='train', offvalue='test')
                    self.DataManagerStorage[index]['checkbox'].grid(row=index+3,column =1,pady=2)
                    # print(self.model.dataset[part]['egzo'][i])
                    text=', '.join([str(foo) for foo in self.model.dataset[part]['egzo'][i]])+' -> '+', '.join([str(foo) for foo in self.model.dataset[part]['endo'][i]])
                    
                    self.DataManagerStorage[index]['label'] = tk.Label(self.DataManagerEditFrame,text =text)
                    self.DataManagerStorage[index]['label'].grid(row=index+3,column =2,pady=2)
                    self.DataManagerStorage[index]['editbtn'] = tk.Button(self.DataManagerEditFrame,text ="Edit",command = lambda index=index: self.DataManagerEdit(index))
                    self.DataManagerStorage[index]['editbtn'].grid(row=index+3,column =3,pady=2)
                    self.DataManagerStorage[index]['delbtn'] = tk.Button(self.DataManagerEditFrame,text='Delete',command=lambda inde = index :self.DataManagerDeleteSubmit(inde))
                    self.DataManagerStorage[index]['delbtn'].grid(row=index+3,column=4)
                    index +=1
            self.DataManagerStorage['index']=index

        def DataManagerEdit(self,index):
            self.DataManagerEditTopLevel={}
            self.DataManagerEditTopLevel[index] = tk.Toplevel()
            self.DataManagerEditTopLevel[index].title('Edit Data')
            self.DataManagerEditEntry={}
            self.DataManagerEditEntry[index]=tk.Entry(self.DataManagerEditTopLevel[index],width = 40)
            self.DataManagerEditEntry[index].grid(row=1,column=0,columnspan = 2)
            value = ', '.join([str(foo) for foo in self.DataManagerStorage[index]['data']['egzo'] ]) + \
                ' -> '+\
                    ', '.join([str(foo) for foo in self.DataManagerStorage[index]['data']['endo'] ])
            self.DataManagerEditEntry[index].insert(0,value)
            self.DataManagerEditSubmitBTN={}
            self.DataManagerEditSubmitDEL={}
            self.DataManagerEditSubmitBTN[index] = tk.Button(self.DataManagerEditTopLevel[index],text='Submit',command=lambda index = index :self.DataManagerEditSubmit(index))
            self.DataManagerEditSubmitBTN[index].grid(row=2,column=0)

        def DataManagerDeleteSubmit(self,index):
            del(self.DataManagerStorage[index])
            print(self.DataManagerStorage.keys(),index)
            self.DataManagerReadAndExit(destroy=False)
            self.DataManagerEditUpdate()

        def DataManagerEditSubmit(self,index):
            value = self.DataManagerEditEntry[index].get()
            if value == '':
                del(self.DataManagerStorage[index])
                self.DataManagerEditTopLevel[index].destroy()
                return 
            try:
                egzo ,endo = value.split('->')
                egzo = [float(i) for i in egzo.split(',')]
                endo = [float(i) for i in endo.split(',')]
                self.DataManagerStorage[index]['data']['egzo'] = egzo
                self.DataManagerStorage[index]['data']['endo'] = endo

            except:
                tkmsg.showwarning("Data Edit Error","Error occured while editing dataset!")
                return
            self.DataManagerReadAndExit(destroy=False)
            self.DataManagerEditUpdate()
            self.DataManagerEditTopLevel[index].destroy()
            # self.DataManagerTopLevel.destroy()

            

        def DataManagerReadAndExit(self,destroy=True):
            self.model.dataset={
                'train':{
                    'egzo':[],
                    'endo':[]
                },
                'test': {
                    'egzo':[],
                    'endo':[]
                }
            }
            for index in self.DataManagerStorage.keys():
                if index == 'index':
                    continue
                place = self.DataManagerStorage[index]['var'].get()
                egzo = self.DataManagerStorage[index]['data']['egzo']
                endo =  self.DataManagerStorage[index]['data']['endo']
                self.model.dataset[
                    place
                ]['egzo'].append(egzo)
                self.model.dataset[
                    place
                ]['endo'].append(endo)
            if destroy:
                self.DataManagerTopLevel.destroy()
        def TrainNetwork(self):
            
            ok=True


            if not self.DataFormatOk:
                if len(self.model.structure) == 0:
                    return
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

            if len(self.model.structure) == 0 or self.model.dataset ==None or self.model.dataset=={} or (not "train" in self.model.dataset.keys()) or (not "train" in self.model.dataset.keys()):
                tkmsg.showerror("Uninitialized Network", "Please make sure the dataset is imported and network is compiled!")
                return 
            self.model.setExternalMonitor(self.TrainNetworkMonitor)
            # TODO add including monitor
            self.model.train()
            try:
                self.TrainStatisticsUpadte()
            except Exception as ex:
                if len(self.LayerMonitorWidgetStorage.keys()) ==0:
                    pass
            tkmsg.showinfo("Model Trained!","Model is trained!")
            self.LayerMonitorWidgetSetupUpdate()
                    
                

        def OneStep(self):
            
            ok=True


            if not self.DataFormatOk:
                if len(self.model.structure) == 0:
                    return
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

            if len(self.model.structure) == 0 or self.model.dataset ==None or self.model.dataset=={} or (not "train" in self.model.dataset.keys()) or (not "train" in self.model.dataset.keys()):
                tkmsg.showerror("Uninitialized Network", "Please make sure the dataset is imported and network is compiled!")
                return 
            self.model.setExternalMonitor(self.TrainNetworkMonitor)
            # TODO add including monitor
            self.model.OneStep()
            try:
                self.TrainStatisticsUpadte()
            except Exception as ex:
                pass
            
            self.LayerMonitorWidgetSetupUpdate()
    

        def Step(self):
            
            ok=True


            if not self.DataFormatOk:
                if len(self.model.structure) == 0:
                    return
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

            if len(self.model.structure) == 0 or self.model.dataset ==None or self.model.dataset=={} or (not "train" in self.model.dataset.keys()) or (not "train" in self.model.dataset.keys()):
                tkmsg.showerror("Uninitialized Network", "Please make sure the dataset is imported and network is compiled!")
                return 
            self.model.setExternalMonitor(self.TrainNetworkMonitor)
            # TODO add including monitor
            self.model.Step()
            try:
                self.TrainStatisticsUpadte()
            except Exception as ex:
                if len(self.LayerMonitorWidgetStorage.keys()) ==0:
                    tkmsg.showinfo("Model Trained!","Model is trained!")
            
            self.LayerMonitorWidgetSetupUpdate()




        def ImportNetworkFromFile(self):
            nameoffile = self.AskForFilePath()
            if nameoffile == () or nameoffile == '':
                return

            with open(nameoffile,'r') as file:
                data = json.load(file)

            try:
                self.model.Import(data['model'])
                self.modelEvaluationTest = data['evaluate_test']
                self.modelEvaluationTrain = data['evaluate_train']
            except Exception as ex:
                tkmsg.showwarning("File Import Failed", "File import failed!")
                print(ex)
                return
            tkmsg.showinfo("Import", "Import done!")
            self.Update()
            return

        def ExportNetworkToFile(self):
            nameeoffile=self.AskSaveAsFileName()
            if nameeoffile == "" or nameeoffile == ():
                return
            temp = {
                'model': self.model.Export(),
                'evaluate_train' : self.modelEvaluationTrain,
                'evaluate_test' : self.modelEvaluationTest
            }
            # print('####\n'+str(temp))
            try:
                with open(nameeoffile,'w+') as file:
                    json.dump(temp,file, cls=NumpyArrayEncoder)
            except Exception as ex:
                print(ex)
                tkmsg.showwarning("File Export Failed", "File export failed!")
                return
            tkmsg.showinfo("Export Done", "Correct Export Data From Network")
            return
        def Evaluate(self):
            if not len(self.model.structure)>0:
                tkmsg.showwarning("Network not compiled","Please compile network to be able to test it on a data!")
                return
            
            self.EvaluateTopLevel = tk.Toplevel()
            self.EvaluateTopLevel.title('Evaluate')

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


        # def DEBUG_helper(self):
            
        #     value = "self.LayerMonitorWidgetStorage"+str(self.LayerMonitorWidgetStorage)+"\n"
        #     value += "self.model.layers"+ str(self.model.layers)+"\n"
        #     value += "self.model.structure"+str(self.model.structure)+"\n"
        #     value += "\n Layers:\n"
        #     for i in self.model.structure:
        #         value += str(i.matrix.shape)+"\n"

        #     value += "Dataset: "+ str(self.model.dataset)+"\n"
        #     self.LabelDEBUG["text"] = value

        def TrainStatisticsUpadte(self):
            try:
                self.TrainStatisticsLabelTrain['text'] = "Minimum Train Error: "+str(np.min(self.modelEvaluationTrain))+\
                    "\nAverage Train Error: "+str(np.average(self.modelEvaluationTrain))+"\nMaximum Train Error: "+str(np.max(self.modelEvaluationTrain))
                self.TrainStatisticsLabelTest['text'] = "Minimum Train Error: "+str(np.min(self.modelEvaluationTrain))+\
                "\nAverage Train Error: "+str(np.average(self.modelEvaluationTrain))+"\nMaximum Train Error: "+str(np.max(self.modelEvaluationTrain))
            except AttributeError as ex:
                pass
        #####
        def TrainStatistics(self):

            if self.modelEvaluationTest is None or len(self.modelEvaluationTest) == 0 or self.modelEvaluationTrain is  None or len(self.modelEvaluationTrain) == 0:
                tkmsg.showinfo("Cannot calculate statistics","Model is not trained or initailized, so it's imposible to calculate values and make charts!")
                return
            self.TrainStatisticsTopLevel = tk.Toplevel()
            self.TrainStatisticsTopLevel.title('Train Statistics')
            
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

            

        def ErrorPlotShow(self):
            plt.plot(self.modelEvaluationTrain)
            plt.plot(self.modelEvaluationTest)
            plt.ylabel( "Error")
            plt.xlabel("Epoch")
            plt.show()

        def ArffDataFormatInport(self,filename,Nendo:int=1):
            dataset={
                'train':{
                    'egzo':[],
                    'endo':[]
                },
                'test':{
                    'egzo':[],
                    'endo':[]
                }
            }

            content = ''
            with open(filename, 'r') as file:
                content = file.read()

            content = content.split('\n')
            toggle = None
            egzo = Nendo
            preprocedDataset = []
            for i in content:
                if re.match('^%.*',i):
                    # print('comment\\\\     '+i)
                    pass
                elif re.match('^((\d*(\.\d*)?),)*\d+(\.\d*)?$',i):
                    preprocedDataset.append(i)
                elif re.match('^@egzogenic *\d+ *$',i):
                    try:
                        egzo=int(re.findall('\d+',i)[0])
                    except Exception as ex:
                        print(ex)
                        pass
                elif re.match('^(@trainData *|@testData *)$',i):
                    preprocedDataset.append(i)
            # print(preprocedDataset,egzo)
            for i in preprocedDataset:
                sp = i.split(' ')[0]
                if sp == '@trainData':
                    toggle="train"
                elif sp == "@testData":
                    toggle='test'
                else:
                    if toggle is None:
                        continue
                    l=i.split(',')
                    l = [float(y) for y in l]
                    dataset[toggle]['egzo'].append(l[:-egzo])
                    dataset[toggle]['endo'].append(l[-egzo:])

            return dataset

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
            elif nameoffile.split('.')[-1] == 'arff':
                self.DataJsonRaw=self.ArffDataFormatInport(nameoffile)
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
            self.CreateNewLayerTopLevel.title('Create New Layer')
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
                self.UpdateNotCompiledStructureWidget()
                self.CreateNewLayerTopLevel.destroy()
            except Exception :
                tkmsg.showerror("Invalid Input!",'Please make sure, number of neurons in this layer is correct!')
                return 
        
        def TrainNetworkMonitor(self,iteration:int = None,wholeError_train:float=0.0,wholeError_test:float=0.0,correction = None):
            self.modelEvaluationTrain.append(float(wholeError_train))
            self.modelEvaluationTest.append(float(wholeError_test))


   
            


        def CompileNetwork(self):
            if not len(self.model.layers)>=2:
                tkmsg.showerror("Too few layers!", "Network requiers at least 2 layers, input and output. Please add them!")
                return
            self.model.compileModel()
            self.Update()
            self.modelEvaluationTrain = []
            self.modelEvaluationTest = []
            # if DEBUG:
            tkmsg.showinfo("compiled!","Model was compiled!")
            
        def Settings(self):
            self.SettingsTopLevel = tk.Toplevel()
            self.SettingsTopLevel.title('Setting')
            self.SettingsLabelMainText = tk.Label(self.SettingsTopLevel,text = "Settings: ")
            self.SettingsLabelMainText.grid(row=1,column=1,sticky="W", columnspan=2)


            self.SettingsCheckBoxBiasVar = tk.IntVar()
            self.SettingsCheckBoxBiasVar.set(1 if self.ModelBiasState else 0)
            
            self.SettingsCheckBoxBias = tk.Checkbutton(self.SettingsTopLevel, text='use Bias',variable=self.SettingsCheckBoxBiasVar, onvalue=1, offvalue=0)
            self.SettingsCheckBoxBias.grid(row=9,column=1,sticky="W",columnspan=2)

            #learning factor
            self.SettingsLearningFactorLabel = tk.Label(self.SettingsTopLevel,text="Learning Factor")
            self.SettingsLearningFactorLabel.grid(row=3,column=1,columnspan=2)

            self.SettingsLearningFactorEntry = tk.Entry(self.SettingsTopLevel)
            self.SettingsLearningFactorEntry.insert(0, str(self.model.learningFactor))
            self.SettingsLearningFactorEntry.grid(row=4,column=1,columnspan=2)
            
            #iterations
            self.SettingsIterationsLabel = tk.Label(self.SettingsTopLevel,text="Iterations: ")
            self.SettingsIterationsLabel.grid(row=5,column=1,columnspan=2)

            self.SettingsIterationsEntry = tk.Entry(self.SettingsTopLevel, width = 20)
            self.SettingsIterationsEntry.insert( 0, str(self.model.iterations))
            self.SettingsIterationsEntry.grid(row=6,column=1,columnspan=2)


            #ErrorCondition
            self.SettingsErrorStopLabel = tk.Label(self.SettingsTopLevel,text="Error value to stop: ")
            self.SettingsErrorStopLabel.grid(row=7,column=1,columnspan=2)

            self.SettingsErrorStopEntry = tk.Entry(self.SettingsTopLevel, width = 20)
            self.SettingsErrorStopEntry.insert( 0, str(self.model.errorValue))
            self.SettingsErrorStopEntry.grid(row=8,column=1,columnspan=2)



            
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
            errorValue = self.SettingsErrorStopEntry.get()
            try:
                errorValue = float(errorValue)
                if errorValue <0:
                    raise Exception()
            except Exception as ex:
                tkmsg.showerror("Wrong Error Value","Wrong Error Value! Please pass correct number!")
                return



            if self.SettingsCheckBoxBiasVar.get()==1:
                self.ModelBiasState = True
            else: 
                self.ModelBiasState = False
            self.model.iterations = iterations
            self.model.learningFactor = learningRate
            self.model.errorValue = errorValue
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

     
