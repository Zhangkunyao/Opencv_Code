import os
import numpy as np
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

def get_bbos(img):
    tmp = (img[:,:,0]>0)*1 + (img[:,:,1]>0)*1 + (img[:,:,2]>0)*1
    tmp = tmp > 0
    sp = tmp.shape  # 行 列
    # 先做二值化处理
    test_scale = 1
    # 思想 选取目标点按以前的方式，暴力外扩 跳变点的方式无法确定是不是同一个物体
    data_map = np.ones(sp)
    glob_bbox = [sp[0], sp[1], 0, 0]
    for hang in range(0, sp[0], int(sp[0] / 100)):
        for lie in range(0, sp[1], int(sp[1] / 100)):
            if data_map[hang][lie] == 1:  # 该点还没被查找过
                if tmp[hang][lie]:  # 该处存在特征点
                    data_map[hang][lie] = 0
                    xmin = lie
                    xmax = lie
                    ymin = hang
                    ymax = hang
                    exit_flag = 0
                    # center=lie   #寻找的中心位置
                    # 先找右边
                    right_hang = hang
                    right_lie = lie
                    left_hang = hang
                    left_lie = lie

                    flag_right_out = 0
                    flag_left_out = 0
                    while exit_flag == 0:
                        flag_right_out = 0
                        flag_left_out = 0
                        # right_lie=center
                        while right_lie + test_scale < sp[1] and tmp[right_hang][right_lie]:  # 找列
                            right_lie = right_lie + test_scale

                        if xmax < right_lie:
                            xmax = right_lie;

                        # 找左边
                        # left_lie = center
                        while left_lie - test_scale > 0 and tmp[left_hang][left_lie]:  # 找列
                            left_lie = left_lie - test_scale

                        if xmin > left_lie:
                            xmin = left_lie
                        # 行标下移
                        if left_hang + test_scale < sp[0]:
                            left_hang = left_hang + test_scale
                        while left_lie + test_scale < right_lie and (tmp[left_hang][left_lie] == False):  # 找行
                            left_lie = left_lie + test_scale
                        if left_lie + test_scale >= right_lie or left_hang + test_scale >= sp[0]:
                            flag_left_out = 1

                        if right_hang + test_scale < sp[0]:
                            right_hang = right_hang + test_scale
                        while right_lie - test_scale > left_lie and (tmp[right_hang][right_lie] == False):  # 找行
                            right_lie = right_lie - test_scale
                        if right_lie - test_scale <= left_lie or right_hang + test_scale >= sp[0]:
                            flag_right_out = 1

                        if (flag_left_out == 1 and flag_right_out == 1):
                            exit_flag = 1

                    if left_hang > right_hang:
                        ymax = left_hang
                    else:
                        ymax = right_hang
                    ymin = hang
                    if xmin < glob_bbox[0]:
                        glob_bbox[0] = xmin
                    if ymin < glob_bbox[1]:
                        glob_bbox[1] = ymin
                    if xmax > glob_bbox[2]:
                        glob_bbox[2] = xmax
                    if ymax > glob_bbox[3]:
                        glob_bbox[3] = ymax
                    data_map[ymin - 1:ymax + 1, xmin - 1:xmax + 1] = 0;
    return {'xmin':glob_bbox[0],'ymin':glob_bbox[1],'xmax':glob_bbox[2],'ymax':glob_bbox[3]}