from training_model import main_training_model
from pre_processing2 import main_pre_processing2_log_close_price
from pre_processing import main_pre_processing_log_close_price
from analyze_model import main_analyze_model
from analyze_model2 import main_analyze_model2

def main():
    """
    preprocess les données data_name et entraine un modele model_name
    """
    N_EPOCHS = 300
    model_name = "test_Close_price_log_jaquart_dense64"
    data_name = "dataset_raw_1d_2017_08_17_2025_01_08"
    full_model_name = model_name+'-'+data_name+'-'+str(N_EPOCHS)

    (date_train, X_train, y_train, date_val, X_val, y_val, date_test, X_test, y_test), scaler_features, scaler_target = main_pre_processing_log_close_price(data_name)
    main_training_model(full_model_name, X_train, y_train, X_val, y_val, N_EPOCHS)
    main_analyze_model(full_model_name)

def main2():
    """
    preprocess2 les données data_name et entraine un modele model_name
    """
    WINDOW_SIZE = 7
    N_EPOCHS = 100
    data_name = "dataset_raw_1d_2017_08_17_2025_01_08"    

    datasets = main_pre_processing2_log_close_price(data_name, WINDOW_SIZE)
    
    model_name = f"preprocess2_Close_price_log_jaquart_dense64_WS{WINDOW_SIZE}"
    full_model_name = model_name+'-'+data_name+'-'+str(N_EPOCHS)

    X_train = datasets['TRAIN'][0][1]
    y_train = datasets['TRAIN'][0][2]
    
    X_val = datasets['VAL'][0][1]
    y_val = datasets['VAL'][0][2]
    
    main_training_model(full_model_name, X_train, y_train, X_val, y_val, N_EPOCHS)
    main_analyze_model2(full_model_name, WINDOW_SIZE)

main2()