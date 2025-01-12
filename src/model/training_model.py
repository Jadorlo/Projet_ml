import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
load_dotenv()


WD = os.getenv('working_directory')
WINDOW_SIZE = 7

def prepare_model(name, n_features):
    """
    Set up le model
    """
    model = Sequential([Input((WINDOW_SIZE, n_features)),
                        LSTM(64),
                        Dense(32, activation='relu'),
                        Dense(32, activation='relu'),
                        Dense(1)
                        ], name=name)
    
    return model

def prepare_checkpoint(name):
    """
    """
    return ModelCheckpoint(f'{WD}/src/model/models/{name}/', save_best_only=True, save_format='tf', monitor='loss')

def save(model):
    """
    """
    model.save(f'{WD}/src/model/models/{model.name}/', save_format='tf')

def train(model, learning_rate):
    """
    """
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])
    return model

def fit(model, cp, X_train, y_train, X_val, y_val, N_EPOCHS):
    """
    """
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=N_EPOCHS, callbacks=[cp])
    return model

def main_training_model(model_name, scaler_features, scaler_target, X_train, y_train, X_val, y_val, N_EPOCHS):

    cp = prepare_checkpoint(name=model_name)
    model = prepare_model(name=model_name,n_features=X_train.shape[2])
    model = train(model, 0.001)
    model = fit(model, cp, X_train, y_train, X_val, y_val, N_EPOCHS)
