
import numpy as np
from PIL import  Image
import ssim
import os

def mse(pic1,pic2):
    pic1 = np.array(pic1)
    pic2 = np.array(pic2)
    cha = np.fabs(pic1-pic2)
    return np.mean(cha)


# for root, dirs, files in os.walk(file_dir):
#     # print(root) #当前目录路径
#     # print(dirs) #当前路径下所有子目录
#     # print(files) #当前路径下所有非目录子文件
#     return len(files)

data_test_path = 'save_result/use_net_person145____use_data_person1'
all = []


list = list(os.listdir(data_test_path))

flames_num = 10


for dir in list:
    this_dic = data_test_path+'/'+dir
    file_res = open(this_dic+'/'+'error.txt','w')

    for i in range(1,flames_num):
        input = Image.open(this_dic+'/'+'input_t_%d.png'%(i))
        pre = Image.open(this_dic + '/' + 'pred_t_%d.png' %(i))

        file_res.write('input%d and pred%d  ssim  is %.4f\n'%(i,i,ssim.compute_ssim(input,pre)))

        file_res.write('input%d and pred%d   mse  is %.4f\n' % (i, i, mse(input, pre)))

    final_flame = Image.open(this_dic + '/' + 'true_next_flame.png')
    final_flame_pred = Image.open(this_dic + '/' + 'pred_t_%d.png' % (flames_num))
    file_res.write('\n')
    file_res.write('final true and pred  ssim  is %.4f\n' % ( ssim.compute_ssim(final_flame, final_flame_pred)))

    file_res.write('final true and pred   mse  is %.4f\n' % ( mse(final_flame, final_flame_pred)))

    file_res.write('\n')
    file_res.write('final true and last_flame  ssim  is %.4f\n' % (ssim.compute_ssim(final_flame, input)))

    file_res.write('final true and last_flame   mse  is %.4f\n' % (mse(final_flame, input)))

    file_res.write('\n')
    file_res.write('final pred and last_flame  ssim  is %.4f\n' % (ssim.compute_ssim(final_flame_pred, input)))

    file_res.write('final pred and last_flame   mse  is %.4f\n' % (mse(final_flame_pred, input)))

# for test_data in all:
