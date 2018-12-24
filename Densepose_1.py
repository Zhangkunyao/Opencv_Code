import numpy
import cv2
import matplotlib.pyplot as plt
from basic_lib import Get_List
import os
import numpy as np
from PIL import Image

def rotate(image, angle, center=None, scale=1.0): #1
    (h, w) = image.shape[:2] #2
    if center is None: #3
        center = (w // 2, h // 2) #4
    M = cv2.getRotationMatrix2D(center, angle, scale) #5
    rotated = cv2.warpAffine(image, M, (w, h)) #6
    return rotated #7

def text_save(filename, data):
    file = open(filename,'a')
    file.write(data)
    file.write('\n')
    file.close()

IUV_path_root = '/media/kun/Dataset/Pose/DataSet/new_data/芭蕾_cut/DensePoseProcess/org'
IUV_save_root = '/media/kun/Dataset/Pose/DataSet/new_data/芭蕾_cut/DensePoseProcess/I'

target_img_path = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/back_ground.png'
target_img = Image.open(target_img_path).convert('RGB')
size_target = target_img.size

_,IUV_ALL = Get_List(IUV_path_root)
IUV_ALL.sort()
index = 0

for name in IUV_ALL:
    IUV_path = os.path.join(IUV_path_root,name)
    IUV_save_path = os.path.join(IUV_save_root,name)

    IUV = Image.open(IUV_path).convert('RGB')
    IUV_size = IUV.size

    scale = 1.0*size_target[1]/IUV.size[1]
    if scale*IUV.size[0]>size_target[0]*2:
        IUV = IUV.resize((int(IUV.size[0]), int(IUV.size[1] * scale)), Image.ANTIALIAS)
    else:
        IUV = IUV.resize((int(IUV.size[0]*scale),int(IUV.size[1]*scale)),Image.ANTIALIAS)
    IUV = np.array(IUV)
    I = IUV[:,:,2]*11
    cv2.imwrite(os.path.join(IUV_save_root,name),I)
    print(name)
    # I = cv2.resize(I, (int(I.shape[1]/2),int(I.shape[0]/2)), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('a',I)
    # cv2.waitKey(0)

# def TransferTexture(TextureIm,im,IUV):
#     U = IUV[:,:,1]
#     V = IUV[:,:,2]
#     #
#     R_im = np.zeros(U.shape)
#     G_im = np.zeros(U.shape)
#     B_im = np.zeros(U.shape)
#     ###
#     for PartInd in xrange(1,25):    ## Set to xrange(1,23) to ignore the face part.
#         PartInd = 3
#         tex = TextureIm[PartInd-1,:,:,:].squeeze() # get texture for each part.
#         #####
#         R = tex[:,:,0]
#         G = tex[:,:,1]
#         B = tex[:,:,2]
#         ###############
#         x,y = np.where(IUV[:,:,0]==PartInd)
#         u_current_points = U[x,y]   #  Pixels that belong to this specific part.
#         v_current_points = V[x,y]
#         ##
#         r_current_points = R[((255-v_current_points)*199./255.).astype(int),(u_current_points*199./255.).astype(int)]*255
#         g_current_points = G[((255-v_current_points)*199./255.).astype(int),(u_current_points*199./255.).astype(int)]*255
#         b_current_points = B[((255-v_current_points)*199./255.).astype(int),(u_current_points*199./255.).astype(int)]*255
#         ##  Get the RGB values from the texture images.
#         R_im[IUV[:,:,0]==PartInd] = r_current_points
#         G_im[IUV[:,:,0]==PartInd] = g_current_points
#         B_im[IUV[:,:,0]==PartInd] = b_current_points
#     generated_image = np.concatenate((B_im[:,:,np.newaxis],G_im[:,:,np.newaxis],R_im[:,:,np.newaxis]), axis =2 ).astype(np.uint8)
#     BG_MASK = generated_image==0
#     generated_image[BG_MASK] = im[BG_MASK]  ## Set the BG as the old image.
#     return generated_image