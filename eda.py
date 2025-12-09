import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

class EDA:
    def __init__(self, df):
        self.df = df
        self.setup_plotting()
    
    def setup_plotting(self):
        """Setup plotting style"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        self.fig_size = (12, 8)
    
    def perform_complete_eda(self):
        """Perform complete exploratory data analysis"""
        print("Performing Exploratory Data Analysis...")
        
        # Basic dataset info
        self.dataset_overview()
        
        # Statistical summary
        self.statistical_summary()
        
        # Visualizations
        self.create_geographic_plot()
        self.create_correlation_analysis()
        self.create_feature_distributions()
        self.create_price_analysis()
        
        # Skip feature relationships if it causes errors, but try it
        try:
            self.create_feature_relationships()
        except Exception as e:
            print(f"Note: Feature relationships plot skipped due to: {e}")
        
        print("EDA complete! Check the results/plots/ directory for visualizations.")
    
    def dataset_overview(self):
        """Display basic dataset information"""
        print("\n" + "="*50)
        print("DATASET OVERVIEW")
        print("="*50)
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Number of Features: {self.df.shape[1]}")
        print(f"Number of Samples: {self.df.shape[0]}")
        print("\nColumn Names and Data Types:")
        print(self.df.dtypes)
        print("\nMissing Values:")
        print(self.df.isnull().sum())
    
    def statistical_summary(self):
        """Display statistical summary"""
        print("\n" + "="*50)
        print("STATISTICAL SUMMARY")
        print("="*50)
        print(self.df.describe())
        
        # Skewness and Kurtosis for numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        print("\nSkewness and Kurtosis:")
        for col in numerical_cols:
            if self.df[col].notna().sum() > 0:
                skewness = stats.skew(self.df[col].dropna())
                kurtosis = stats.kurtosis(self.df[col].dropna())
                print(f"{col}: Skewness={skewness:.3f}, Kurtosis={kurtosis:.3f}")
    
    def create_geographic_plot(self):
        """Create geographic distribution plot"""
        if 'longitude' in self.df.columns and 'latitude' in self.df.columns and 'median_house_value' in self.df.columns:
            plt.figure(figsize=self.fig_size)
            scatter = plt.scatter(self.df['longitude'], self.df['latitude'], 
                                c=self.df['median_house_value'], 
                                cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label='Median House Value')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title('Geographic Distribution of House Prices in California')
            plt.tight_layout()
            plt.savefig('results/plots/geographic_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved geographic distribution plot")
    
    def create_correlation_analysis(self):
        """Create correlation matrix and heatmap"""
        numerical_df = self.df.select_dtypes(include=[np.number])
        
        if len(numerical_df.columns) > 1:
            # Correlation matrix
            correlation_matrix = numerical_df.corr()
            
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, square=True, fmt='.2f')
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig('results/plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Print top correlations with target
            if 'median_house_value' in correlation_matrix.columns:
                target_correlations = correlation_matrix['median_house_value'].sort_values(ascending=False)
                print("\nTop Correlations with Median House Value:")
                print(target_correlations.head(10))
            print("Saved correlation matrix")
    
    def create_feature_distributions(self):
        """Create distribution plots for all features"""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            # Create subplots for numerical features
            n_cols = 3
            n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows > 1:
                axes = axes.flatten()
            else:
                axes = [axes] if n_cols == 1 else axes
            
            for i, col in enumerate(numerical_cols):
                if i < len(axes):
                    self.df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
            
            # Hide empty subplots
            for i in range(len(numerical_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('results/plots/feature_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved feature distributions plot")
        
        # Categorical features
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            plt.figure(figsize=(10, 6))
            value_counts = self.df[col].value_counts()
            bars = plt.bar(value_counts.index, value_counts.values)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f'results/plots/{col}_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {col} distribution plot")
    
    def create_price_analysis(self):
        """Create analysis focused on the target variable"""
        if 'median_house_value' in self.df.columns:
            # Price distribution
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            self.df['median_house_value'].hist(bins=50, alpha=0.7)
            plt.title('Distribution of Median House Values')
            plt.xlabel('House Value ($)')
            plt.ylabel('Frequency')
            
            plt.subplot(1, 2, 2)
            stats.probplot(self.df['median_house_value'], dist="norm", plot=plt)
            plt.title('Q-Q Plot of House Values')
            
            plt.tight_layout()
            plt.savefig('results/plots/price_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved price analysis plot")
    
    def create_feature_relationships(self):
        """Create scatter plots showing relationships with target"""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Select top 6 features most correlated with target
        if 'median_house_value' in self.df.columns and len(numerical_cols) > 1:
            correlations = self.df[numerical_cols].corr()['median_house_value'].abs().sort_values(ascending=False)
            top_features = correlations.index[1:7]  # Exclude target itself
            
            n_cols = 3
            n_rows = 2
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, feature in enumerate(top_features):
                if i < len(axes):
                    # Clean the data - remove any infinite or problematic values
                    clean_data = self.df[[feature, 'median_house_value']].replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if len(clean_data) > 0:
                        axes[i].scatter(clean_data[feature], clean_data['median_house_value'], alpha=0.5)
                        axes[i].set_xlabel(feature)
                        axes[i].set_ylabel('Median House Value')
                        axes[i].set_title(f'Price vs {feature}')
                        
                        # Add trend line with error handling
                        try:
                            if len(clean_data) > 1:
                                z = np.polyfit(clean_data[feature], clean_data['median_house_value'], 1)
                                p = np.poly1d(z)
                                x_range = np.linspace(clean_data[feature].min(), clean_data[feature].max(), 100)
                                axes[i].plot(x_range, p(x_range), "r--", alpha=0.8)
                        except Exception as e:
                            print(f"Could not add trend line for {feature}: {e}")
                    else:
                        axes[i].text(0.5, 0.5, f'No valid data for {feature}', 
                                   ha='center', va='center', transform=axes[i].transAxes)
                        axes[i].set_title(f'Price vs {feature} (No Data)')
            
            # Hide empty subplots
            for i in range(len(top_features), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('results/plots/feature_relationships.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved feature relationships plot")