import os

path='C:\\Users\\Rivaille\\Desktop\\ROkinect\\dataset3\\li\\li_test\\temp3'
keyword='_0_'
rekeyword='_180_'

def getPath(path):
    if path.strip()!='':
        try:
            os.chdir(path)
        except os.error:
            print (path+":No such dir...")
    else:
        path=os.getcwd()
    return path

def findFiles(path,keyword):
    all_files=os.listdir(path)
    files=[]
    for filename in all_files:
        if os.path.isfile(path+'\\'+filename) and filename.find(keyword)!=-1:
            files.append(filename)
    return files

def replaceKeyword(fiels,keyword,rekeyword):
    refiles=[]
    for filename in files:
        refiles.append(filename.replace(keyword,rekeyword,1))

    return refiles

def renameFiles(files,refiles,path):

    for i in range(len(files)):
        try:
            os.rename(path+'\\'+files[i],path+'\\'+refiles[i])
        except os.error:
            print(path+'\\'+files[i]+':wrong')

path=getPath(path)
files=findFiles(path,keyword)
refiles=replaceKeyword(files,keyword,rekeyword)
renameFiles(files,refiles,path)