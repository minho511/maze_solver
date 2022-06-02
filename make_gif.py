import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import PIL
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def mkgif():
    path = [f"./gifs/{i}" for i in os.listdir("./gifs")]
    path.sort()
    paths = [ Image.open(i) for i in path]
    print(len(paths))
    imageio.mimsave('./result.gif', paths, fps=20)
    img = Image.open('./result.gif')
    img.show()