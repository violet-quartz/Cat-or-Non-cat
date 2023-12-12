import numpy as np
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import sklearn.metrics
import joblib
import os
import yaml

train_dir = os.path.dirname(__file__)
train_dataset_rel_path = os.path.join('datasets', 'train_catvnoncat.h5')
test_dataset_rel_path = os.path.join('datasets', 'test_catvnoncat.h5')

def load_dataset():
    train_dataset = h5py.File(os.path.join(train_dir, train_dataset_rel_path), "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    # print(train_set_x_orig.shape)
    # print(train_set_y_orig.shape)

    test_dataset = h5py.File(os.path.join(train_dir, test_dataset_rel_path), "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    # print(test_set_x_orig.shape)
    # print(test_set_y_orig.shape)

    classes = np.array(test_dataset["list_classes"][:])
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
print(classes)
print(f'Number of training examples is {train_set_x_orig.shape[0]}')
print(f'Number of test examples is {test_set_x_orig.shape[0]}')
image_height, image_width = train_set_x_orig.shape[1], train_set_x_orig.shape[2]
print(f'Each image is of shape: ({train_set_x_orig.shape[1]}, {train_set_x_orig.shape[2]}, {train_set_x_orig.shape[3]}) '
      + f'where 3 is for 3 channels (RGB). Image height={image_height}, width={image_width}')

# Flatten the image into a single vector
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1)
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1)

# Standardize the dataset
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

train_set_y = train_set_y.flatten()
test_set_y = test_set_y.flatten()
print(f'train_set_x shape: {train_set_x_flatten.shape}')
print(f'train_set_y shape: {train_set_y.shape}.')

# train
# TODO: improve model's performance
model = LogisticRegression()
param_grid = {'C': [0.01, 0.1, 1], 'max_iter': [1000, 5000, 10000]}
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1')
grid_search.fit(train_set_x, train_set_y)

best_params = grid_search.best_params_
print(f'best_params: {best_params}')
model = LogisticRegression(**best_params)
model.fit(train_set_x, train_set_y)

# test
predictions = model.predict(test_set_x)
f1_score = sklearn.metrics.f1_score(test_set_y, predictions)
print(f'precision={sklearn.metrics.precision_score(test_set_y, predictions)}')
print(f'recall={sklearn.metrics.recall_score(test_set_y, predictions)}')


# save the model to a file
model_name = 'model.joblib'
model_path = os.path.join(os.path.dirname(train_dir), 'model', model_name)
joblib.dump(model, model_path)

# save the config
data = {
    'train_dataset': train_dataset_rel_path,
    'test_dataset': test_dataset_rel_path,
    'model_type': 'sklearn.linear_model.LinearRegression',
    'image_height': image_height,
    'image_width': image_width,
    'model_rel_path': model_name
}
config_file = os.path.join(os.path.dirname(train_dir), 'model', 'config.yaml')
with open(config_file, 'w') as f:
    yaml.dump(data, f)

print(f'Save config to {config_file}')
print(f'Save model to {model_path}')






