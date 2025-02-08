import qadence
import torch
import numpy as np
from load import load_txt

# estimate the value used to generate the data in data/dataset_1_a.txt
def estimate_value():
    # Load the data
    x, y = load_txt("../data/dataset_1_a.txt")
    print(x)
    print(y)


estimate_value()

