from PIL import Image

im1 = Image.open("lena.png")
im2 = Image.open("lena_modified.png")
#im.show()

width, height = im1.size
print(width)
print(height)


for y in range(height):
    for x in range(width):
        rgba1 = im1.getpixel( (x, y) )
        rgba2 = im2.getpixel( (x, y) )
        #print(rgba)
        if rgba1 == rgba2 :
            rgba2 = (rgba2[0] - rgba1[0],  # R
                     rgba2[1] - rgba1[1],  # G
                     rgba2[2] - rgba1[2],  # B
                     rgba2[3] - rgba1[3]);  # A
        im1.putpixel((x, y), rgba2)

im1.show()
im1.save("new.png")


