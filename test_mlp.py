import numpy as np
import pandas as pd
import pickle
from train_mlp import MLP_Q4

STUDENT_NAME = 'YOUR NAME'
STUDENT_ID = 'YOUR_ID'

def test_mlp(data_file):
    # Load the test set

    # START
    test_set = pd.read_csv(data_file).values
    # END

    # Load your network
    # START
    with open('mlp_params.pickle', 'rb') as handle:
        params = pickle.load(handle)
    mlp_model = MLP_Q4(params['hidden_nodes'], params)
    o, _ = mlp_model.forward(test_set)
    # END

    # Predict test set - one-hot encoded # y_pred = ...
    y_hat = np.argmax(o, axis=0)
    n_values = 4
    y_pred = np.eye(n_values)[y_hat]

    return y_pred

'''
How we will test your code:

from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy 

y_pred = test_mlp('./test_data.csv')

test_labels = ...

test_accuracy = accuracy(test_labels, y_pred)*100
'''
