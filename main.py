import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# IMPORTS - ABSOLUTE IMPORTS WITHOUT DOTS
try:
    from data_loader import DataLoader
    from eda import EDA
    from model_trainer import ModelTrainer
    from utils import ResultsVisualizer
except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure these files are in the same directory as main.py:")
    print("- data_loader.py")
    print("- eda.py") 
    print("- model_trainer.py")
    print("- utils.py")
    print("- config.py")
    print("\nCurrent directory files:")
    for file in os.listdir('.'):
        if file.endswith('.py'):
            print(f"  - {file}")
    exit(1)

def create_directories():
    """Create necessary directories"""
    directories = [
        'data/raw',
        'data/processed', 
        'data/external',
        'models',
        'results/plots',
        'results/metrics',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Created project directory structure")

def main():
    """Main execution function"""
    print("Real Estate Price Prediction Project")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    try:
        # Step 1: Load data
        print("\nStep 1: Loading Data...")
        data_loader = DataLoader()
        housing_df = data_loader.load_housing_data()
        
        # Save raw data
        housing_df.to_csv('data/raw/housing_data.csv', index=False)
        print(f"Housing data shape: {housing_df.shape}")
        print(f"Columns: {list(housing_df.columns)}")
        
        # Step 2: Perform EDA
        print("\nStep 2: Performing Exploratory Data Analysis...")
        eda = EDA(housing_df)
        eda.perform_complete_eda()
        
        # Step 3: Train models
        print("\nStep 3: Training Machine Learning Models...")
        trainer = ModelTrainer(housing_df)
        trainer.preprocess_data()
        trainer.train_models()
        
        # Step 4: Evaluate models
        print("\nStep 4: Evaluating Models...")
        results_df = trainer.evaluate_models()
        best_model, best_model_name = trainer.get_best_model()
        
        # Step 5: Create visualizations
        print("\nStep 5: Creating Results Visualizations...")
        visualizer = ResultsVisualizer(trainer)
        visualizer.create_all_visualizations()
        
        # Step 6: Save results
        print("\nStep 6: Saving Results...")
        results_df.to_csv('results/metrics/model_performance.csv')
        
        # Save feature importance
        if trainer.feature_importance:
            for model_name, importance_df in trainer.feature_importance.items():
                importance_df.to_csv(f'results/metrics/feature_importance_{model_name}.csv', index=False)
        
        # Final summary
        print("\n" + "="*60)
        print("PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Best Model: {best_model_name}")
        print(f"Best R² Score: {trainer.results[best_model_name]['R2']:.4f}")
        print(f"Best MAE: ${trainer.results[best_model_name]['MAE']:,.2f}")
        print(f"Best RMSE: ${trainer.results[best_model_name]['RMSE']:,.2f}")
        print(f"\nResults saved in 'results/' directory")
        print("Visualizations saved in 'results/plots/' directory")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        print("Please check that all required files are present and config.py is correct.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()