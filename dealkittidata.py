
import numpy as np

from PIL import Image

pic = np.load('kittidata/test.npy')
picsouce  = np.load('kittidata/soureces_test.npy')

# for i in range( len(pic)):
# #     ima = Image.fromarray(pic[i],'RGB')
# #     ima.save('kittidata/pic/'+str(i)+'.png')
# #     # ima.show()
this  = picsouce[0]
x = []
len_flames=11
for i in range(len(pic)-len_flames):
    if picsouce[i]==picsouce[i+len_flames]:
        this_flames = pic[i:i+len_flames]
        x.append(this_flames)

x = np.array(x)

np.save('kittidata/test_flames.npy',x)


    # if picsouce[i] !=this:
    #     print(i)
    #     this = picsouce[i]