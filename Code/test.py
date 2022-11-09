import ora
from torch.utils.data import Dataset

import utils
from utils import make_one_hot
from PIL import Image

print("Hello world")
#image = ora.read_ora('my_file.ora')
#000 je originalni ortopanogram
#sve ostalo su segmentacijske karte, u ovoj m   api NEMAMO potpuni ortopanogram ali mislim da nam i ne treba
# print(image['root']['childs'].__len__())
# for i in range (len(image['root']['childs'])):
#     print(image['root']['childs'][i])
# print(image['root']['childs'][0])
#print("Hello world")

img = Image.open("./patient001_frame01_2_gt.png")
img2 = utils.make_one_hot(img, "acdc", 0).float()
print(img2)
