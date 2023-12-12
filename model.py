import os
import joblib
from PIL import Image
import numpy as np
import yaml
import matplotlib.pyplot as plt

class CatPictureDetectModel:
    def __init__(self, model_config):
        with open(model_config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.image_height = config['image_height']
        self.image_width = config['image_width']
        self.model = joblib.load(os.path.join(os.path.dirname(model_config), config['model_rel_path']))
        
    def predict(self, img_path) -> bool:
        """
        Return True if it's a cat picture, else False.
        """
        # Preprocess the image to fit the model
        image = np.array(Image.open(img_path).resize((self.image_height, self.image_width)))
        print(image.shape)
        plt.imshow(image)
        image = image.reshape((1, self.image_height * self.image_width * 3))
        image = image / 255
        # Predict
        label = self.model.predict(image)
        return label[0] == 1 
        