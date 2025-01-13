from training_model import main_training_model
from pre_processing import main_pre_processing

def main():
    """
    preprocess les données data_name et entraine un modele model_name
    """
    N_EPOCHS = 100

    model_name = "Close_price"
    data_name = "dataset_raw_6h"
    full_model_name = model_name+'-'+data_name+'-'+str(N_EPOCHS)
    (date_train, X_train, y_train, date_val, X_val, y_val, date_test, X_test, y_test), scaler_features, scaler_target = main_pre_processing(data_name, is_one_var=True)
    main_training_model(full_model_name, scaler_features, scaler_target, X_train, y_train, X_val, y_val, N_EPOCHS)

main()