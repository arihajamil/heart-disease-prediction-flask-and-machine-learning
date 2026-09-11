"""
Research and training package
==============================
Everything used to BUILD the models, kept separate from the web
application that serves them.

    heart.csv         the raw dataset (70,000 patient records)
    train_models.py   the 13-step preprocessing and training pipeline

The web app never needs this package to serve predictions - it only loads
the finished .pkl files from models/. The one exception is the admin
"Model Training" page, which imports train_models to rebuild them on demand.
"""
