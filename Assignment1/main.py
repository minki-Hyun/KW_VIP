# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from PIL import Image

image = Image.open("./lenna.png")

# show image
image.show()

# reverse image
image_reverse = image.transpose(Image.FLIP_LEFT_RIGHT)

# show reverse image
image_reverse.show()

# rotate image
image_rotate = image.transpose(Image.ROTATE_180)

# show rotate image
image_rotate.show()

# resize image
image_resize_half = image.resize((256, 256))

# show resize image
image_resize_half.show()
