# run.py - Simple runner that will definitely work
import os
import sys

print("Starting Real Estate Price Prediction Project")

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import all required modules
    from data_loader import DataLoader
    from eda import EDA 
    from model_trainer import ModelTrainer
    from utils import ResultsVisualizer
    
    print("All imports successful!")
    
    # Create directories
    directories = ['data/raw', 'data/processed', 'results/plots', 'results/metrics']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    data_loader = DataLoader()
    housing_df = data_loader.load_housing_data()
    print(f"Loaded data: {housing_df.shape}")
    
    # Perform EDA
    print("\nPerforming EDA...")
    eda = EDA(housing_df)
    eda.perform_complete_eda()
    
    # Train models
    print("\nTraining models...")
    trainer = ModelTrainer(housing_df)
    trainer.preprocess_data()
    trainer.train_models()
    
    # Show results
    print("\nModel Results:")
    trainer.evaluate_models()
    
    print("\nProject completed successfully!")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Files in current directory:")
    for file in os.listdir('.'):
        print(f"  - {file}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()