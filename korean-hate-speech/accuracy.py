import os
import shutil
import time
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


test = pd.read_csv('/root/team26/DeepOffense/examples/korean/temp/data/result.csv', sep="\t")

label = test['labels'].to_numpy()
pred = test['predictions'].to_numpy()
acc = accuracy_score(label,pred)
f1 = f1_score(label,pred,average='macro')
w_f1 = f1_score(label,pred,average='weighted')
print("accuracy: ",acc)
print("f1_score: ",f1)
print("weighted_f1: ",w_f1)
