import multiprocessing
import numpy as np
import csv

def replace_null_values_with_mean(X):
    #Obtain mean of columns
    col_mean = np.nanmean(X, axis=0)

    #Find indicies that we need to replace
    inds = np.where(np.isnan(X))

    #Place column means in the indices. Align the arrays using take
    X[inds] = np.take(col_mean, inds[1])
    return X

def mean_normalize(X, column_indices):
    for column_index in column_indices:
        column = X[:,column_index]
        min = np.min(column, axis=0) 
        max = np.max(column, axis=0)
        avg = np.average(column, axis=0)
        difference = max- min
        X[:,column_index] = (column - avg) /difference
        mmm = np.array([avg, min, max])
        save_model2(mmm, 'MMM_file.csv')
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

def import_data():
    X = np.genfromtxt("train_X_pr.csv", delimiter=',', dtype=np.float64, skip_header=1)
    Y = np.genfromtxt("train_Y_pr.csv", delimiter=',', dtype=np.float64)
    X = replace_null_values_with_mean(X)
    X = mean_normalize(X, [2, 5])
    X = convert_given_cols_to_one_hot(X, [0, 3])
    #X = X[0:int(0.75*len(X))]
    #Y = Y[0:int(0.75*len(Y))]
    return X, Y

def sigmoid(Z):
    s = 1 / (1 + np.exp(-Z))
    return s

def compute_cost(X, Y, W, b, Lambda):
    M = len(X)
    Z = np.dot(X, W) + b
    A = sigmoid(Z)
    cost = (-1/M) * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A)) 
    regularization_cost = (Lambda * np.sum(np.square(W))) / (2 * M) 
    return cost + regularization_cost

def compute_gradients_using_regularization(X, Y, W, b, Lambda):
    m = len(X)
    Z = np.dot(X, W) + b
    A = sigmoid(Z)
    dW = 1/m * (np.dot((A-Y).T, X) + Lambda*(W.T))
    db = 1/m * np.sum(A-Y)
    dW = dW.T
    return dW, db

def optimize_weights_using_gradient_descent(X, Y, W, b, learning_rate, Lambda):
    previous_iter_cost = 0
    iter_no = 0
    while True:
        iter_no += 1
        dW, db = compute_gradients_using_regularization(X, Y, W, b, Lambda)
        W = W - (learning_rate * dW)
        b = b - (learning_rate * db)
        cost = compute_cost(X, Y, W, b, Lambda)
        if abs(previous_iter_cost - cost) < 0.000001:
            print(iter_no, cost)
            break
        #if iter_no % 1000 == 0:
        #print(iter_no, cost)
        previous_iter_cost = cost
    return W, b

def train_model(X, Y):
    W = np.ones((X.shape[1], ))
    b = 1
    W, b = optimize_weights_using_gradient_descent(X, Y, W, b, 1, 10.5)
    W = np.append(W, b)
    return W

def save_model(weights, weights_file_name):
    with open(weights_file_name, 'w', newline='') as weights_file:
        wr = csv.writer(weights_file)
        wr.writerows(map(lambda x: [x], weights))
        weights_file.close()
        
def save_model2(weights, weights_file_name):
    with open(weights_file_name, 'a', newline='') as weights_file:
        wr = csv.writer(weights_file)
        wr.writerow(weights)
        weights_file.close()

if __name__ == "__main__":
    X, Y = import_data()
    weights = train_model(X, Y)
    save_model(weights, "WEIGHTS_FILE.csv")
