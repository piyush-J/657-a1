from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy
import pandas as pd

y_pred = test_mlp("train_data.csv")

test_labels = pd.read_csv("train_labels.csv").values

test_accuracy = accuracy(test_labels, y_pred)*100

print(test_accuracy)