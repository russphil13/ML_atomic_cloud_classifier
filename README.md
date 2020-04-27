# Atomic Cloud Classifier
ML analysis for images of atomic clouds

This is a project for teaching myself to use popular Python ML libraries. Simulated training data is created from statistical analysis of images of clouds of atomic gases from ultracold atoms experiments and used to train a binary classifier. The classifier is applied to actual experiment images and acts as a thredhold function for determining if an atomic cloud is present or not. Experiment images and data are stored on a local PostgreSQL server.

##Libraries
 - Python 3.7
 - Matplotlib 3.1.3
 - NumPy 1.18.1
 - Psycopg 2.8.4
 - scikit-learn 0.22.1
 - XGBoost 1.0.2

##Usage
###Tuning hyper-parameters
Use script tune\_(model-type)\_model.py to find the optimal hyper-parameters.

###Predictions on atomic clouds.
Use the script (model-type)\_binary\_classify.py to train the model using the optimal hyper-parameters and get the predictions on the atomic cloud dataset. The standard analysis of counting pixels is plotted against the analysis using the binary classifier.
