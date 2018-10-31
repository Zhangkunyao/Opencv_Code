import cv2
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

root_path = '/media/kun/UbuntuData/Kun/Dance/Dance-Basic-D-Match/results'
_,img_all = Get_List(root_path)
img_all.sort()
img = cv2.imread(os.path.join(root_path,img_all[0]))
fps = 30
img_size = (img.shape[1],img.shape[0])
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter('result.avi', fourcc, fps, img_size)
if not videoWriter.isOpened():
    print("video error")
    exit(0)

for i in range(len(img_all)):
    img = cv2.imread(os.path.join(root_path, img_all[i]))
    videoWriter.write(img)
    print(1.0*i/len(img_all))
videoWriter.release()
print('finish')


