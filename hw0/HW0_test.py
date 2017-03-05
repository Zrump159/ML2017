import random
import sys
import os
import numpy as np
from PIL import Image

matrix_f = open ("matrixB.txt" , "r")


matrixA = []
for line in matrix_f :
    line = line.strip()
    print('first for loop read each line:'+line)
    #strip removed some character from beginning or end of the string.
    for a in line.split(',') :
        #use split to cut line to each character
        #ex: [5,8,6] become [ '5' , ',' , '8' , ',' , '6' ]
        print('second for loop read each character:'+a)

im = Image.open("lena.png")
#im.show()

width, height = im.size
print(width)
print(height)

data=im.getdata()

for y in range(height):
    for x in range(width):
        rgba = im.getpixel((x, y))
        print(rgba)
        rgba = (255 - rgba[0],  # R
                255 - rgba[1],  # G
                255 - rgba[2],  # B
                rgba[3]);  # A
        im.putpixel((x, y), rgba)

im.show()
im.save("new.png")