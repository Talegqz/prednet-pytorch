from PIL import  Image
import numpy  as np


my_pred = Image.open('save_result/pred1.png')
mybum = np.array(my_pred)
mybum = mybum/255
source = Image.open(r'D:\database\action1-person1-white\frame\1042.png')
source = np.array(source)
source = source/255

a = np.sum(mybum-source)


data = np.load('D:/database/action1-person1-white/prednetflames12.npy')

one  = data[0][1]
one = np.transpose(one, (1, 2, 0))
this_image = Image.fromarray(one, 'RGB')


this_image.show()


print(a)

print('')