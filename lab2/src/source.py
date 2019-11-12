# A little introductory for you, folks.
# We are writing a program to solve multi-classification problem
# https://www.kaggle.com/c/cifar-10/overview dataset from here
# PLEASE, PAY ATTENTION!
# If you will create a folder for new lab, don't forget to include venv and .idea and another system
# directories to .gitignore

import glob
from PIL import Image
import numpy as np


def get_cifar_data(path='C:\\Users\\Максим\\Desktop\\Test Data for lab2\\train\\train\\*.png', count=10):
    filelist = glob.glob(path)[:count]
    x = np.array([np.array(Image.open(fname)) for fname in filelist])
    return x