import pandas as pd
import numpy as np
import requests
from io import StringIO
import os

try:
    from .config import Config
except ImportError:
    from config import Config

class DataLoader:
    def __init__(self):
        self.config = Config()
        
    def load_housing_data(self):
        """Load California Housing Prices dataset"""
        print("Loading California Housing Prices dataset...")
        try:
            # Load from GitHub
            url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
            housing_df = pd.read_csv(url)
            print(f"Loaded housing data: {housing_df.shape}")
            return housing_df
        except Exception as e:
            print(f"Error loading housing data: {e}")
            print("Using fallback data...")
            return self._generate_fallback_data()
    
    def load_economic_data(self):
        """Load economic indicators"""
        print("Loading economic indicators...")
        try:
            # Generate synthetic economic data based on housing data years
            years = range(1990, 2023)
            economic_data = {
                'year': list(years),
                'gdp_growth': np.random.normal(2.5, 1.5, len(years)),
                'inflation_rate': np.random.normal(2.0, 0.8, len(years)),
                'employment_rate': np.random.normal(95, 2, len(years)),
                'interest_rate': np.random.normal(4.5, 2, len(years))
            }
            return pd.DataFrame(economic_data)
        except Exception as e:
            print(f"Error loading economic data: {e}")
            return None
    
    def load_demographic_data(self):
        """Load demographic data for California regions"""
        print("Loading demographic data...")
        try:
            # Create demographic data based on California regions
            regions = ['North Coast', 'San Francisco Bay', 'Central Coast', 
                      'Sacramento Valley', 'San Joaquin Valley', 'Southern California']
            
            demographic_data = {
                'region': regions,
                'population_density': np.random.randint(50, 5000, len(regions)),
                'median_income': np.random.randint(30000, 120000, len(regions)),
                'crime_rate': np.random.uniform(0.5, 8.0, len(regions)),
                'education_index': np.random.uniform(0.6, 0.95, len(regions)),
                'unemployment_rate': np.random.uniform(3.0, 12.0, len(regions))
            }
            return pd.DataFrame(demographic_data)
        except Exception as e:
            print(f"Error loading demographic data: {e}")
            return None
    
    def _generate_fallback_data(self):
        """Generate fallback data if online loading fails"""
        print("Generating fallback data...")
        np.random.seed(42)
        n_samples = 2000
        
        data = {
            'longitude': np.random.uniform(-124.3, -114.3, n_samples),
            'latitude': np.random.uniform(32.5, 42.0, n_samples),
            'housing_median_age': np.random.randint(1, 52, n_samples),
            'total_rooms': np.random.randint(2, 4000, n_samples),
            'total_bedrooms': np.random.randint(1, 650, n_samples),
            'population': np.random.randint(3, 3600, n_samples),
            'households': np.random.randint(1, 620, n_samples),
            'median_income': np.random.uniform(0.5, 15.0, n_samples),
            'ocean_proximity': np.random.choice(['INLAND', 'NEAR BAY', '<1H OCEAN', 'NEAR OCEAN'], n_samples),
        }
        
        # Create realistic price based on features
        base_price = (
            data['median_income'] * 50000 +
            (52 - data['housing_median_age']) * 1000 +
            data['total_rooms'] * 10 +
            np.random.normal(0, 50000, n_samples)
        )
        data['median_house_value'] = np.maximum(base_price, 50000)
        
        return pd.DataFrame(data)