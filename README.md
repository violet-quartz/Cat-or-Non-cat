# Cat or Non-cat
A web application where you can upload an image and it will judge whether it's a cat picture or not. The web framework used is [flask](https://flask.palletsprojects.com/en/3.0.x/). The model used for binary classification(cat or not cat) is Logistic Regression.
## Use the application
Run the application locally.
```
> pip install requirements.txt
> python app.py
```
Open the generated link in web browser. Choose image file and upload it, then the judge result will be returned.

<img width="500" alt="image_cat" src="https://github.com/violet-quartz/Cat-or-Non-cat/assets/79560376/6a20e489-87a3-44b2-a8b2-a4c50871cef7" >

<img width="500" alt="image_non_cat" src="https://github.com/violet-quartz/Cat-or-Non-cat/assets/79560376/3249160c-8b15-4c74-a733-e73691a868fd">

## Model training
[train.py](.\train\train.py) displays the training process.  
[LogisticRegression.py](.\train\LogisticRegression.py) implement the logistic regression algorithm.


