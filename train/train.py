import numpy as np
import h5py
import os
import yaml
from LogisticRegression import LogisticRegression
import sklearn.metrics
import matplotlib.pyplot as plt

train_dir = os.path.dirname(__file__)
# Replace the following ones with your train/test dataset path.
train_dataset_rel_path = os.path.join('datasets', 'train_catvnoncat.h5')
test_dataset_rel_path = os.path.join('datasets', 'test_catvnoncat.h5')

def load_dataset():
    train_dataset = h5py.File(os.path.join(train_dir, train_dataset_rel_path), "r")
    train_set_x_orig = np.array(train_dataset['train_set_x'][:]) # your train set features
    train_set_y_orig = np.array(train_dataset['train_set_y'][:]) # your train set labels
    # print(train_set_x_orig.shape)
    # print(train_set_y_orig.shape)

    test_dataset = h5py.File(os.path.join(train_dir, test_dataset_rel_path), "r")
    test_set_x_orig = np.array(test_dataset['test_set_x'][:]) # your test set features
    test_set_y_orig = np.array(test_dataset['test_set_y'][:]) # your test set labels
    # print(test_set_x_orig.shape)
    # print(test_set_y_orig.shape)

    classes = np.array(test_dataset['list_classes'][:])
    
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
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# Standardize the dataset
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

print(f'train_set_x shape: {train_set_x_flatten.shape}')
print(f'train_set_y shape: {train_set_y.shape}.')

# train
model = LogisticRegression()
num_iterations = 2000
learning_rate=0.005
params, grads, costs = model.fit(train_set_x, train_set_y, 
                             num_iter=num_iterations, learning_rate=learning_rate, print_cost=True, cost_interval=100)
w, b = params['w'], params['b']

# predict
train_set_y_predict = model.predict(w, b, train_set_x)
test_set_y_predict = model.predict(w, b, test_set_x)

print(f'train accuracy: {sklearn.metrics.accuracy_score(np.squeeze(train_set_y), np.squeeze(train_set_y_predict))}')
print(f'test accuracy: {sklearn.metrics.accuracy_score(np.squeeze(test_set_y), np.squeeze(test_set_y_predict))}')
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations per hundreds')
plt.show()

# save the model to a file
model_name = 'model.h5'
model_path = os.path.join(os.path.dirname(train_dir), 'model', model_name)
with h5py.File(model_path, 'w') as model_file:
    model_file.create_dataset('weights', data=w)
    model_file.create_dataset('bias', data=b)

# save the config
data = {
    'train_dataset': train_dataset_rel_path,
    'test_dataset': test_dataset_rel_path,    
    'image_height': image_height,
    'image_width': image_width,
    'model_rel_path': model_name,
    'model_type': 'LogisticRegression',
    'num_iterations': num_iterations,
    'learning_rate': learning_rate
}
config_file = os.path.join(os.path.dirname(train_dir), 'model', 'config.yaml')
with open(config_file, 'w') as f:
    yaml.dump(data, f)

print(f'Save config to {config_file}')
print(f'Save model to {model_path}')






