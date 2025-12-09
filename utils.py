import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class ResultsVisualizer:
    def __init__(self, model_trainer):
        self.trainer = model_trainer
        self.results = model_trainer.results
        self.feature_importance = model_trainer.feature_importance
        self.setup_plotting()
    
    def setup_plotting(self):
        """Setup plotting style"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        self.fig_size = (12, 8)
    
    def create_all_visualizations(self):
        """Create all result visualizations"""
        self.plot_model_comparison()
        self.plot_feature_importance()
        self.plot_predictions_vs_actual()
        self.plot_residual_analysis()
    
    def plot_model_comparison(self):
        """Create model performance comparison plots"""
        if not self.results:
            print("No results to visualize")
            return
        
        results_df = pd.DataFrame(self.results).T
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # R² Comparison
        results_df['R2'].plot(kind='bar', ax=axes[0,0], color='lightgreen')
        axes[0,0].set_title('R² Score Comparison')
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # MAE Comparison
        results_df['MAE'].plot(kind='bar', ax=axes[0,1], color='lightcoral')
        axes[0,1].set_title('Mean Absolute Error (MAE) Comparison')
        axes[0,1].set_ylabel('MAE ($)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # RMSE Comparison
        results_df['RMSE'].plot(kind='bar', ax=axes[1,0], color='lightblue')
        axes[1,0].set_title('Root Mean Squared Error (RMSE) Comparison')
        axes[1,0].set_ylabel('RMSE ($)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Cross-validation R²
        results_df['CV_R2_Mean'].plot(kind='bar', ax=axes[1,1], color='gold', 
                                     yerr=results_df['CV_R2_Std'], capsize=4)
        axes[1,1].set_title('Cross-Validation R² Score (Mean ± Std)')
        axes[1,1].set_ylabel('CV R² Score')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('results/plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved model comparison plot")
    
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        if not self.feature_importance:
            print("No feature importance data available")
            return
        
        n_models = len(self.feature_importance)
        if n_models == 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
        
        for idx, (model_name, importance_df) in enumerate(self.feature_importance.items()):
            if idx < len(axes):
                top_features = importance_df.head(10)
                if n_models == 1:
                    axes.barh(top_features['feature'], top_features['importance'])
                    axes.set_title(f'Top 10 Features - {model_name}')
                    axes.set_xlabel('Importance')
                else:
                    axes[idx].barh(top_features['feature'], top_features['importance'])
                    axes[idx].set_title(f'Top 10 Features - {model_name}')
                    axes[idx].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('results/plots/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved feature importance plot")
    
    def plot_predictions_vs_actual(self):
        """Plot predictions vs actual values for all models"""
        if not self.results:
            return
        
        n_models = len(self.results)
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            if n_cols == 1:
                axes = [axes]
            else:
                axes = axes
        else:
            axes = axes.flatten()
        
        for idx, (model_name, metrics) in enumerate(self.results.items()):
            if idx < len(axes):
                y_pred = metrics['predictions']
                axes[idx].scatter(self.trainer.y_test, y_pred, alpha=0.5)
                axes[idx].plot([self.trainer.y_test.min(), self.trainer.y_test.max()], 
                              [self.trainer.y_test.min(), self.trainer.y_test.max()], 'r--', lw=2)
                axes[idx].set_xlabel('Actual Prices')
                axes[idx].set_ylabel('Predicted Prices')
                axes[idx].set_title(f'{model_name}\nR² = {metrics["R2"]:.4f}')
        
        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('results/plots/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved predictions vs actual plot")
    
    def plot_residual_analysis(self):
        """Plot residual analysis for the best model"""
        if not self.results or self.trainer.best_model_name is None:
            return
        
        best_model_name = self.trainer.best_model_name
        y_pred = self.results[best_model_name]['predictions']
        residuals = self.trainer.y_test - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title(f'Residuals vs Predicted - {best_model_name}')
        
        # Residuals distribution
        axes[1].hist(residuals, bins=30, alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Residuals Distribution - {best_model_name}')
        
        plt.tight_layout()
        plt.savefig('results/plots/residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved residual analysis plot")