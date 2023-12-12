import numpy as np
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    # print(train_set_x_orig.shape)
    # print(train_set_y_orig.shape)

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    # print(test_set_x_orig.shape)
    # print(test_set_y_orig.shape)

    classes = np.array(test_dataset["list_classes"][:])
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
print(f'Number of training examples is {train_set_x_orig.shape[0]}')
print(f'Number of test examples is {test_set_x_orig.shape[0]}')
print(f'Each image is of size: ({train_set_x_orig.shape[1]}, {train_set_x_orig.shape[2]}, {train_set_x_orig.shape[3]})')

# Flatten the image into a single vector
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1)
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1)

# Standardize the dataset
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

train_set_y = train_set_y.flatten()
test_set_y = test_set_y.flatten()
print(f'train_set_x shape: {train_set_x_flatten.shape}')
print(f'train_set_y shape: {train_set_y.shape}')

# train
# TODO: tune parameters and split validation set to pick one best model.
model = LogisticRegression(max_iter=1000)
model.fit(train_set_x, train_set_y)

# test
predictions = model.predict(test_set_x)
accuracy = accuracy_score(test_set_y, predictions)
print(f'Accuracy: {accuracy}')

# save the model to a file
# TODO: update path dir
joblib.dump(model, '..\models\model.joblib')







