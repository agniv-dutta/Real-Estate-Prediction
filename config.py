# Configuration settings
class Config:
    # Random seed for reproducibility
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    
    # Data URLs
    HOUSE_PRICES_URL = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
    ECONOMIC_INDICATORS_URL = "https://raw.githubusercontent.com/datasets/consumer-price-index/main/data/current.csv"
    
    # Model parameters
    RF_PARAMS = {
        'n_estimators': 200,
        'random_state': RANDOM_STATE,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    }
    
    XGB_PARAMS = {
        'n_estimators': 200,
        'random_state': RANDOM_STATE,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }