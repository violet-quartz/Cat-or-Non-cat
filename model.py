import os
import h5py
from PIL import Image
import numpy as np
import yaml
import matplotlib.pyplot as plt
from train.LogisticRegression import LogisticRegression

class CatPictureDetectModel:
    def __init__(self, model_config):
        with open(model_config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.image_height = config['image_height']
        self.image_width = config['image_width']
        # load model parameters
        self.model = LogisticRegression()
        model_path = os.path.join(os.path.dirname(model_config), config['model_rel_path'])
        with h5py.File(model_path, 'r') as model_file:
            self.w =model_file['weights'][()]
            self.b = model_file['bias'][()]      
        
    def predict(self, img_path) -> bool:
        """
        Return True if it's a cat picture, else False.
        """
        # Preprocess the image to fit the model
        image = np.array(Image.open(img_path).resize((self.image_height, self.image_width)))
        print(image.shape)
        plt.imshow(image)
        image = image.reshape((self.image_height * self.image_width * 3, 1))
        image = image / 255
        # Predict
        label = self.model.predict(self.w, self.b, image)[0]
        return label == 1 
        