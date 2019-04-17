import matplotlib.pyplot as plt
import matplotlib.collections as mcol
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
from matplotlib.lines import Line2D
import numpy as np
from basic_lib import Get_List
import os

def takefirst(elem):
    return elem[0]

def get_txt_data(path):
    file = open(path, 'r')
    listall = file.readlines()
    if ':' in listall[0]:
        listall = [i.rstrip().split(':') for i in listall]
    else:
        listall = [i.rstrip('\n').split('\t') for i in listall]
    data = []
    for i in listall:
        data.append([int(i[0]),float(i[1])])
    data.sort(key=takefirst)

    file.close()
    return data

line_style = [':|','-|','--|','-.|',':x',':x','-x','--x','-.x',':x']
txt_path = '/media/kun/Dataset/GAN_Relation/celeba_64/txt'
_,txt_all = Get_List(txt_path)

name_all = []
line_all = []
fig, ax = plt.subplots()
min_length = 100000000
for name in txt_all:
    if name[-3:] != 'txt':
        continue
    line_name = name[:-4]
    name_all.append(line_name)
    data = get_txt_data(os.path.join(txt_path,name))
    line_all.append(data)
    if len(data)<min_length:
        min_length = len(data)

line_all_plot = []
index = 0
for data in line_all:
    data = np.array(data)
    l, = ax.plot(data[:,0],data[:,1],line_style[index])
    line_all_plot.append(l)
    print(name_all[index])
    index += 1

ax.legend(line_all_plot, name_all, loc='upper right', shadow=True)
ax.set_xlabel('Generator Iteration')
ax.set_ylabel('FID Score')
plt.savefig(os.path.join(txt_path,'fid.png'))
plt.show()
