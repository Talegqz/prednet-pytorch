import torch.nn as nn
import torch
import numpy as np
# rnn = nn.LSTM(10, 20, 1)
# input = torch.randn(5, 3, 10)
# h0 = torch.randn(2, 3, 10)
# c0 = torch.randn(2, 3, 10)
# output, hn = rnn(input, (h0, c0))
#


import ssim
a = np.arange(0,45).reshape(3,3,5)
c = np.arange(45,90).reshape(3,3,5)

afe = ssim.compute_ssim(a,c)

print(afe)




print('')