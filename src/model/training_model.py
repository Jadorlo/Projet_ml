import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Input, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
load_dotenv()


WD = os.getenv('working_directory')


def prepare_model(name, n_features, window_size):
    """
    Set up le model
    """
    model = Sequential(
                        [Input((window_size, n_features)),
                        LSTM(256),
                        Dropout(0.5),
                        Dense(64, activation='relu'),
                        Dropout(0.2),
                        # Dense(32, activation='relu'),
                        Dense(1)
                        ], name=name)
    
    return model


def prepare_model_bayesian(name, n_features, window_size):
    """
    Bayesian Neural Network Model
    """
    # Input Layer
    inputs = Input((window_size, n_features))
    
    # LSTM Layer
    lstm_out = LSTM(256)(inputs)
    
    # Dropout Layer
    lstm_out = Dropout(0.5)(lstm_out)
    
    # Bayesian Dense Layer 1
    dense_bayesian_1 = tfpl.DenseVariational(
        units=64,
        activation='relu',
        make_prior_fn=tfpl.default_multivariate_normal_fn,
        make_posterior_fn=tfpl.default_mean_field_normal_fn,
        kl_weight=1/window_size,  # KL divergence regularization
    )(lstm_out)
    
    # Dropout Layer
    dense_bayesian_1 = Dropout(0.2)(dense_bayesian_1)
    
    # Bayesian Dense Output Layer
    outputs = tfpl.DenseVariational(
        units=1,
        make_prior_fn=tfpl.default_multivariate_normal_fn,
        make_posterior_fn=tfpl.default_mean_field_normal_fn,
        kl_weight=1/window_size,
    )(dense_bayesian_1)
    
    # Model
    model = Model(inputs=inputs, outputs=outputs, name=name)
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

def main_training_model(model_name, X_train, y_train, X_val, y_val, window_size, N_EPOCHS):

    cp = prepare_checkpoint(name=model_name)
    model = prepare_model(name=model_name,n_features=X_train.shape[2], window_size=window_size)
    model = train(model, 0.001)
    model = fit(model, cp, X_train, y_train, X_val, y_val, N_EPOCHS)

def main_training_model_bayesian(model_name, X_train, y_train, X_val, y_val, window_size, N_EPOCHS):

    cp = prepare_checkpoint(name=model_name)
    model = prepare_model_bayesian(name=model_name,n_features=X_train.shape[2], window_size=window_size)
    model = train(model, 0.001)
    model = fit(model, cp, X_train, y_train, X_val, y_val, N_EPOCHS)
