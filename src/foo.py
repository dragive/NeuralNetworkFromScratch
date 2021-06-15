import re

from numpy import csingle
def ArffDataFormatInport(filename,Nendo:int=1):
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
#     content = '''%comment

# %comment 2
# @egzogenic 1
# @trainData
  
# 1,1,1,1,0.5
# 1,1,0,0,1
# 1,1,1,0,0

# @testData
# 2,3,4,5,6'''
    
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
ArffDataFormatInport('a')