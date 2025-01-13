from training_model import main_training_model
from pre_processing import main_pre_processing_pourcentage, main_pre_processing_close_price, main_pre_processing_log_close_price
from analyze_model import main_analyze_model

def main():
    """
    preprocess les données data_name et entraine un modele model_name
    """
    N_EPOCHS = 300
    model_name = "test_Close_price_log_jaquart_dense64"
    data_name = "dataset_raw_1d_2017_08_17_2025_01_08"
    full_model_name = model_name+'-'+data_name+'-'+str(N_EPOCHS)

    (date_train, X_train, y_train, date_val, X_val, y_val, date_test, X_test, y_test), scaler_features, scaler_target = main_pre_processing_log_close_price(data_name)
    main_training_model(full_model_name, scaler_features, scaler_target, X_train, y_train, X_val, y_val, N_EPOCHS)
    main_analyze_model(full_model_name)
main()