import numpy as np
import numpy.linalg as LA
from PIL import Image
import matplotlib.pyplot as plt
##loading face

data = [[0]*64*64]*100
data = np.asarray(data,np.int)
w = ['A','B','C','D','E','F','G','H','I','J']
recon_face = Image.new('L',(64,64))

for i in range(10) :
    for j in range(10) :
        tmp = Image.open("face/%c%02d.bmp"%(w[i],j))
        pix = np.array(list(tmp.getdata()))
        data[10 * i + j] = pix   #100*4096
data = data.transpose() #4096*100
conv_matrix = np.cov(data)
e_value,e_vector = LA.eig(conv_matrix)
e_vector = e_vector.transpose()
data = data.transpose() #100*4096

top5 = np.argsort(e_value)[-9:]
tmp = Image.open("avg.bmp")
pix = np.array(list(tmp.getdata()))

for i in top5 :
    lenght = np.sum(np.multiply(pix, e_vector[i]))
    tmp = np.multiply(lenght, e_vector[i])
    tmp = np.array(np.rint(np.abs(tmp)), np.int)
    tmp = tmp.reshape((64, 64))
    plt.subplot(3, 3, i + 1)
    plt.axis('off')
    plt.imshow(tmp, cmap="gray")
plt.show()


'''
for i in range(100) :
    print(i)
    tmp = [0.]*4096
    for j in top5 :
        lenght = np.sum(np.multiply((data[i]-pix),e_vector[j]))
        re_vector = np.multiply(lenght,e_vector[j])
        tmp += re_vector
    tmp += pix
    tmp = np.array(np.rint(np.abs(tmp)),np.int)
    # recon_face.putdata(tmp)
    tmp = tmp.reshape((64, 64))
    plt.subplot(10,10,i+1)
    plt.axis('off')
    plt.imshow(tmp, cmap="gray")
plt.show()
'''