import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

try:
    from .config import Config
except ImportError:
    from config import Config

class ModelTrainer:
    def __init__(self, df):
        self.df = df
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self.best_model = None
        self.config = Config()
        
    def preprocess_data(self):
        """Preprocess the data for modeling"""
        print("Preprocessing data...")
        
        # Create a copy of the dataframe
        df_processed = self.df.copy()
        
        # Handle missing values
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        # Fill numerical missing values with median
        for col in numerical_cols:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
        
        # Feature engineering
        df_processed = self.feature_engineering(df_processed)
        
        # Separate features and target
        if 'median_house_value' in df_processed.columns:
            X = df_processed.drop('median_house_value', axis=1)
            y = df_processed['median_house_value']
        else:
            raise ValueError("Target column 'median_house_value' not found in dataset")
        
        # Identify categorical and numerical columns
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create preprocessing pipelines
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_STATE
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def feature_engineering(self, df):
        """Create new features from existing ones"""
        print("Performing feature engineering...")
        
        # Create new features
        if 'total_rooms' in df.columns and 'households' in df.columns:
            df['rooms_per_household'] = df['total_rooms'] / df['households']
        
        if 'total_bedrooms' in df.columns and 'total_rooms' in df.columns:
            df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
        
        if 'population' in df.columns and 'households' in df.columns:
            df['population_per_household'] = df['population'] / df['households']
        
        if 'median_income' in df.columns:
            df['income_squared'] = df['median_income'] ** 2
        
        # Create location-based features
        if 'longitude' in df.columns and 'latitude' in df.columns:
            df['distance_from_coast'] = np.sqrt(
                (df['longitude'] + 119) ** 2 + (df['latitude'] - 37) ** 2
            )
        
        # Create interaction terms
        if 'median_income' in df.columns and 'housing_median_age' in df.columns:
            df['income_age_interaction'] = df['median_income'] * df['housing_median_age']
        
        print(f"New dataset shape after feature engineering: {df.shape}")
        return df
    
    def train_models(self):
        """Train multiple machine learning models"""
        print("\nTraining machine learning models...")
        
        # Define models with pipelines
        models = {
            'Linear Regression': Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', LinearRegression())
            ]),
            'Ridge Regression': Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', Ridge(alpha=1.0, random_state=self.config.RANDOM_STATE))
            ]),
            'Lasso Regression': Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', Lasso(alpha=0.1, random_state=self.config.RANDOM_STATE))
            ]),
            'Random Forest': Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', RandomForestRegressor(**self.config.RF_PARAMS))
            ]),
            'XGBoost': Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', XGBRegressor(**self.config.XGB_PARAMS))
            ]),
            'Gradient Boosting': Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=self.config.RANDOM_STATE))
            ])
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train model
                model.fit(self.X_train, self.y_train)
                
                # Make predictions
                y_pred = model.predict(self.X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(self.y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
                r2 = r2_score(self.y_test, y_pred)
                
                # Cross-validation scores
                cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                          cv=self.config.CV_FOLDS, scoring='r2')
                
                # Store results
                self.models[name] = model
                self.results[name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'CV_R2_Mean': cv_scores.mean(),
                    'CV_R2_Std': cv_scores.std(),
                    'predictions': y_pred
                }
                
                # Extract feature importance for tree-based models
                if hasattr(model.named_steps['regressor'], 'feature_importances_'):
                    # Get feature names after preprocessing
                    feature_names = self.get_feature_names(model.named_steps['preprocessor'])
                    importances = model.named_steps['regressor'].feature_importances_
                    
                    self.feature_importance[name] = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False).head(15)
                
                print(f"{name} - R2: {r2:.4f}, MAE: ${mae:,.2f}, RMSE: ${rmse:,.2f}")
                print(f"Cross-validation R2: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        # Identify best model
        if self.results:
            self.best_model_name = max(self.results.items(), key=lambda x: x[1]['R2'])[0]
            self.best_model = self.models[self.best_model_name]
            print(f"\nBest Model: {self.best_model_name}")
    
    def get_feature_names(self, column_transformer):
        """Get feature names after preprocessing"""
        feature_names = []
        
        # Numerical features
        if 'num' in column_transformer.named_transformers_:
            num_features = column_transformer.named_transformers_['num'].named_steps['scaler'].get_feature_names_out()
            feature_names.extend(num_features)
        
        # Categorical features
        if 'cat' in column_transformer.named_transformers_:
            cat_features = column_transformer.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
            feature_names.extend(cat_features)
        
        return feature_names
    
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        if not self.results:
            print("No models to evaluate. Please train models first.")
            return
        
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*60)
        
        # Create results dataframe
        results_df = pd.DataFrame(self.results).T
        results_df = results_df[['MAE', 'RMSE', 'R2', 'CV_R2_Mean', 'CV_R2_Std']]
        
        print("\nModel Performance Comparison:")
        print(results_df.round(4))
        
        return results_df
    
    def get_best_model(self):
        """Get the best performing model"""
        if self.best_model is None:
            print("No models trained yet.")
            return None, None
        
        best_results = self.results[self.best_model_name]
        print(f"\nBest Model: {self.best_model_name}")
        print(f"R² Score: {best_results['R2']:.4f}")
        print(f"MAE: ${best_results['MAE']:,.2f}")
        print(f"RMSE: ${best_results['RMSE']:,.2f}")
        
        return self.best_model, self.best_model_name