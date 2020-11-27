import numpy as np
import csv
import sys

#from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path', using the weights learned during training.
Writes the predicted values to the file named "predicted_test_Y_pr.csv".
"""

def replace_null_values_with_mean(X):
    #Obtain mean of columns
    col_mean = np.nanmean(X, axis=0)

    #Find indicies that we need to replace
    inds = np.where(np.isnan(X))

    #Place column means in the indices. Align the arrays using take
    X[inds] = np.take(col_mean, inds[1])
    return X

def mean_normalize(X, column_indices, mmm):
    i = 0
    for column_index in column_indices:
        column = X[:,column_index]
        X[:,column_index] = (column - mmm[i,0]) / (mmm[i, 2] - mmm[i, 1])
        i += 1
    return X

def apply_one_hot_encoding(X):
    X = np.floor(X)
    one_hot_encoding_map = {}
    counter = 0
    for x in list(set(X)):
        one_hot_encoding_map[x] = [0 for i in range(len(list(set(X))))]
        one_hot_encoding_map[x][counter] = 1
        counter += 1

    one_hot_encoded_X = []
    for x in X:
        one_hot_encoded_X.append(one_hot_encoding_map[x])

    one_hot_encoded_X = np.array(one_hot_encoded_X, dtype=int)
    return one_hot_encoded_X

def convert_given_cols_to_one_hot(X, column_indices):
    one_hot_encoded_X = np.zeros([len(X),1])

    start_index = 0
    #acts column pointer in X

    for curr_index in column_indices:
        #adding the columns present before curr_index in X (and not present in one_hot_encoded_X), to one_hot_encoded_X
        one_hot_encoded_X=np.append(one_hot_encoded_X,X[:, start_index:curr_index], axis=1)
        
        #applying one hot encoding for current column
        one_hot_encoded_column = apply_one_hot_encoding(X[:,curr_index])

        #appending the obtained one hot encoded array to one_hot_encoded_X
        one_hot_encoded_X=np.append(one_hot_encoded_X,one_hot_encoded_column, axis=1)

        #moving the column pointer of X to next current_index
        start_index = curr_index+1

    #adding any remaining columns to one_hot_encoded_X    
    one_hot_encoded_X=np.append(one_hot_encoded_X,X[:,start_index:], axis=1)
    one_hot_encoded_X = one_hot_encoded_X[:,1:]
    return one_hot_encoded_X

def import_data_and_weights(test_X_file_path, weights_file_path, mmm_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    weights = np.genfromtxt(weights_file_path, delimiter=',', dtype=np.float64)
    mmm = np.genfromtxt(mmm_file_path, delimiter=',', dtype=np.float64)
    test_X = replace_null_values_with_mean(test_X)
    test_X = mean_normalize(test_X, [2, 5], mmm)
    test_X = convert_given_cols_to_one_hot(test_X, [0, 3])
    return test_X, weights

def sigmoid(Z):
    s = 1 / (1 + np.exp(-Z))
    return s

def predict_target_values(test_X, weights):
    test_Y = []
    for x in test_X:
        h = sigmoid(np.dot(x, np.array(weights[:len(x)])) + weights[len(x)])
        if h >= 0.5:
            test_Y.append(1)
        else:
            test_Y.append(0)
    return np.array(test_Y)

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, weights = import_data_and_weights(test_X_file_path, "WEIGHTS_FILE.csv", "MMM_file.csv")
    #test_X = test_X[int(0.75*len(test_X)):]
    pred_Y = predict_target_values(test_X, weights)
    write_to_csv_file(pred_Y, "predicted_test_Y_pr.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    #validate(test_X_file_path, actual_test_Y_file_path="train_Y_pr.csv") 
