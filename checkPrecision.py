import os
import pandas
import numpy

def loadPredictData(filePath):
    data={}
    with open(filePath,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            parts=line.split(",")
            if len(parts)!=2:
                continue
            imgName=parts[0].strip()
            imgName=imgName[:-4]
            plate=parts[1].strip()
            data[imgName]=plate
    return data
def loadData(filePath):
    data = {}
    with open(filePath,"r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line :
                continue
            parts = line.split(",")
            if(len(parts)!=2):
                continue
            imgName=parts[0].strip()
            plate=parts[1].strip()
            data[imgName]=plate
    
    return data

def check(filePath1, filePath2):
    realData=loadData(filePath1)
    predData=loadPredictData(filePath2)
    total, correct, wrong, notfound=0,0,0,0
    wrong_list=[]
    for imgName, realPlate in realData.items():
        total+=1
        if imgName not in predData:
            notfound+=1
            wrong_list.append(f"{imgName},real:{realPlate}:NOT FOUND")
            continue
        predPlate=predData[imgName]
        if realPlate == predPlate:
            correct+=1
        else:
            wrong+=1
            wrong_list.append(f"{imgName},real:{realPlate}:WWRONG")
    return total, correct, wrong, notfound

t, c, w, n = check("fileRealPlate.txt", "results3.txt")
print("total:",t)
print("correct:",c)
print("wrong:",w)
print("notfound:",n)