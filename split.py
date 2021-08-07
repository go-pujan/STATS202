import numpy as np
import pandas as pd
import code

data = pd.read_csv("train_data.csv")
alphabetList = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

for alphabet in alphabetList:
    sliceData = data[data["symbol"] == alphabet]
    sliceData.to_csv(alphabet + "_train_data.csv")
