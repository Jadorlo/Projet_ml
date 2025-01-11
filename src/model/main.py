from training_model import main_training_model
from pre_processing import main_pre_processing

def main():
    """
    """
    model_name = input("Entre le nom du modèle : ")
    data_name = input("Entre le nom du fichier de données : ")
    (date_train, X_train, y_train, date_val, X_val, y_val, date_test, X_test, y_test), scaler = main_pre_processing(data_name)
    main_training_model(model_name, X_train, y_train, X_val, y_val)


main()