# 🏠 Real Estate Price Prediction (California Housing)

A complete **end-to-end machine learning project** for predicting California housing prices using exploratory data analysis, advanced preprocessing, feature engineering, and multiple regression models.  
The goal is to build accurate, interpretable, and scalable models to estimate **median house values** based on geographic, demographic, and housing-related features.

## 📊 Dataset Overview
This project uses the **California Housing Prices Dataset**, containing:

- **20,640 samples**
- **9 predictive features**
- **Target variable:** `median_house_value`

### Key Features  
- **longitude, latitude** – Geographic coordinates  
- **housing_median_age** – Median age of houses  
- **total_rooms, total_bedrooms** – Housing size indicators  
- **population, households** – Demographic metrics  
- **median_income** – Strong economic predictor  
- **ocean_proximity** – Categorical location descriptor  

## 🚀 Project Features

### 1. Exploratory Data Analysis (EDA)
- Geographic distribution maps  
- Correlation heatmaps  
- Feature distribution plots  
- Q-Q plots for price analysis  
- Scatter plots for relationship insights  

### 2. Data Preprocessing
- Handling missing values  
- Creation of **6 new engineered features**  
- One-Hot Encoding for categorical variables  
- Standardization with **StandardScaler**  
- 80-20 train/test split  

### 3. Machine Learning Models
Includes training and comparison of:

- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- Random Forest Regressor  
- XGBoost Regressor  
- Gradient Boosting Regressor  

### 4. Model Evaluation Metrics
- **R² Score**  
- **Mean Absolute Error (MAE)**  
- **Root Mean Squared Error (RMSE)**  
- **5-Fold Cross-Validation**  

### 5. Visualization Outputs
- Geographic price heatmaps  
- Correlation matrices  
- Feature importance charts  
- Actual vs predicted plots  
- Model performance comparison  

## 📁 Project Structure

```
real-estate-prediction/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/
├── notebooks/
├── results/
│   ├── plots/
│   └── metrics/
├── src/
│   ├── data_loader.py
│   ├── eda.py
│   ├── model_trainer.py
│   ├── utils.py
│   └── config.py
├── main.py
├── requirements.txt
└── README.md
```

## 🛠️ Installation

### Prerequisites
- Python **3.8+**
- `pip` package manager

### Setup Steps

```bash
git clone <repository-url>
cd real-estate-prediction
```

(Optional) Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scriptsctivate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 📈 Usage

### Run the Complete ML Pipeline

```bash
python main.py
```

This will:
- Load dataset  
- Preprocess & engineer features  
- Perform EDA  
- Train 6 ML models  
- Evaluate model performance  
- Generate & save visualizations  

## 📊 Key Results

### Top Predictive Features
- `median_income`  
- `distance_from_coast`  
- `rooms_per_household`  
- `bedrooms_per_room`  
- `longitude`, `latitude`  

## 🔧 Configuration
Modify `config.py` to adjust:
- Train/test split  
- Random seed  
- Cross-validation folds  
- Hyperparameters  

## 🤝 Contributing
- Fork repository  
- Create feature branch  
- Commit changes  
- Open Pull Request  

## 📄 License
For educational and research purposes.
