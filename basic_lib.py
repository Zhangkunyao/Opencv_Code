import os

def Get_List(path):
    files = os.listdir(path);
    dirList = []
    fileList = []
    for f in files:
        if (os.path.isdir(path + '/' + f)):
            if (f[0] == '.'):
                pass
            else:
                dirList.append(f)
        if (os.path.isfile(path + '/' + f)):
            fileList.append(f)
    return [dirList, fileList]

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
