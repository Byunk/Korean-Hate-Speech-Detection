import os
import shutil
import time
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

train2 = pd.read_csv('/root/team26/DeepOffense/examples/korean/data/labeled_data.csv', sep=",")
train2 = train2.rename(columns={'tweet': 'text', 'class': 'labels'})
train2 = train2[['text', 'labels']]

# text = train2['text']

for i in range(len(train2['text'])):
    temp= train2['text'][i].strip('"')
    temp = " ".join(filter(lambda x:x[0]!='@', temp.split()))
    temp = " ".join(filter(lambda x:x[0]!='&', temp.split()))
    temp = " ".join(filter(lambda x:x[0:4]!='http', temp.split()))
    temp = " ".join(filter(lambda x:x[0:2]!='RT', temp.split()))
    train2.loc[i, 'text'] = temp
print(train2)