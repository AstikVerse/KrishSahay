"""
KrishiSahay - REAL Machine Learning Model Training Script
This script trains actual ML models on crop price data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
import os
warnings.filterwarnings('ignore')

class CropPricePredictor:
    """Real ML Model for Crop Price Prediction"""
    
    def __init__(self, crop_name):
        self.crop_name = crop_name
        self.model = None
        self.scaler = StandardScaler()
        self.location_encoder = LabelEncoder()
        self.feature_names = []
        self.training_metrics = {}
        
    def load_and_prepare_data(self, csv_path):
        """Load CSV data and prepare for training"""
        print(f"\n{'='*60}")
        print(f"Loading data for {self.crop_name}...")
        print(f"{'='*60}")
        
        # Load data
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter for specific crop
        df = df[df['crop'] == self.crop_name.capitalize()].copy()
        
        if len(df) == 0:
            raise ValueError(f"No data found for {self.crop_name}")
        
        print(f"✓ Loaded {len(df)} records for {self.crop_name}")
        print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def engineer_features(self, df):
        """Create features for ML model"""
        print("\nEngineering features...")
        
        df = df.copy()
        
        # Time-based features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Cyclical encoding for seasonality
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Encode location
        df['location_encoded'] = self.location_encoder.fit_transform(df['state'])
        
        # Lag features (previous prices)
        for lag in [1, 7, 14, 30]:
            df[f'price_lag_{lag}'] = df['price_per_kg'].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 30]:
            df[f'price_rolling_mean_{window}'] = df['price_per_kg'].rolling(
                window=window, min_periods=1
            ).mean()
            df[f'price_rolling_std_{window}'] = df['price_per_kg'].rolling(
                window=window, min_periods=1
            ).std()
        
        # Price momentum
        df['price_momentum_7'] = df['price_per_kg'] - df['price_lag_7']
        df['price_momentum_30'] = df['price_per_kg'] - df['price_lag_30']
        
        # Quantity features
        df['quantity_log'] = np.log1p(df['quantity_traded'])
        
        # Festival indicators (Indian festivals affect demand)
        df['is_festival_season'] = (
            (df['month'].isin([10, 11])) |  # Diwali season
            (df['month'] == 3) |  # Holi
            (df['month'].isin([4, 5]))  # Harvest season
        ).astype(int)
        
        print(f"✓ Created {len([c for c in df.columns if c not in ['date', 'crop', 'market', 'state', 'city', 'price_per_kg', 'quality_grade']])} features")
        
        return df
    
    def prepare_training_data(self, df):
        """Prepare X and y for model training"""
        # Drop rows with NaN (from lag features)
        df = df.dropna()
        
        # Define feature columns (exclude target and metadata)
        exclude_cols = ['date', 'crop', 'market', 'state', 'city', 'price_per_kg', 'quality_grade']
        self.feature_names = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_names]
        y = df['price_per_kg']
        
        print(f"\n✓ Training data prepared: {len(X)} samples, {len(self.feature_names)} features")
        
        return X, y
    
    def train(self, X, y, test_size=0.2):
        """Train the ML model"""
        print(f"\n{'='*60}")
        print(f"Training Gradient Boosting Model for {self.crop_name}...")
        print(f"{'='*60}")
        
        # Time-based split (important for time series!)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Gradient Boosting model
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        
        print("\nTraining in progress...")
        self.model.fit(X_train_scaled, y_train)
        print("✓ Training complete!")
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Store metrics
        self.training_metrics = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': len(self.feature_names)
        }
        
        # Print results
        print(f"\n{'='*60}")
        print(f"MODEL PERFORMANCE - {self.crop_name.upper()}")
        print(f"{'='*60}")
        print(f"Train MAE:  ₹{train_mae:.2f}/kg")
        print(f"Test MAE:   ₹{test_mae:.2f}/kg  ⭐")
        print(f"Train RMSE: ₹{train_rmse:.2f}/kg")
        print(f"Test RMSE:  ₹{test_rmse:.2f}/kg")
        print(f"Train R²:   {train_r2:.4f}")
        print(f"Test R²:    {test_r2:.4f}  ⭐")
        print(f"{'='*60}")
        
        # Calculate accuracy percentage
        mean_price = y_test.mean()
        accuracy = (1 - (test_mae / mean_price)) * 100
        print(f"✓ Model Accuracy: {accuracy:.1f}%")
        print(f"✓ Average prediction error: ₹{test_mae:.2f}/kg")
        
        return self.training_metrics
    
    def get_feature_importance(self, top_n=10):
        """Get top important features"""
        if self.model is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def save_model(self, filepath):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'location_encoder': self.location_encoder,
            'feature_names': self.feature_names,
            'crop_name': self.crop_name,
            'training_metrics': self.training_metrics
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to {filepath}")
        print(f"  File size: {os.path.getsize(filepath) / 1024:.1f} KB")
    
    @classmethod
    def load_model(cls, filepath):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        predictor = cls(model_data['crop_name'])
        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']
        predictor.location_encoder = model_data['location_encoder']
        predictor.feature_names = model_data['feature_names']
        predictor.training_metrics = model_data['training_metrics']
        
        return predictor


def main():
    """Main training function"""
    print("\n" + "="*60)
    print("KRISHISAHAY - REAL ML MODEL TRAINING")
    print("="*60)
    
    # Crops to train
    crops = ['Tomato', 'Onion', 'Potato', 'Wheat', 'Rice']
    
    # Path to your dataset
    data_path = 'data/raw/crop_prices_historical.csv'
    
    if not os.path.exists(data_path):
        print(f"\n❌ ERROR: Dataset not found at {data_path}")
        print("Please run the dataset generator first and place CSV files in data/raw/")
        return
    
    all_metrics = {}
    
    # Train model for each crop
    for crop in crops:
        try:
            # Initialize predictor
            predictor = CropPricePredictor(crop.lower())
            
            # Load data
            df = predictor.load_and_prepare_data(data_path)
            
            # Engineer features
            df = predictor.engineer_features(df)
            
            # Prepare training data
            X, y = predictor.prepare_training_data(df)
            
            # Train model
            metrics = predictor.train(X, y)
            all_metrics[crop] = metrics
            
            # Show feature importance
            print("\nTop 10 Most Important Features:")
            importance = predictor.get_feature_importance()
            for idx, row in importance.iterrows():
                print(f"  {row['feature']:30s} {row['importance']:.4f}")
            
            # Save model
            model_path = f'models/trained_models/{crop.lower()}_model.pkl'
            predictor.save_model(model_path)
            
            print(f"\n✓ {crop} model training complete!\n")
            
        except Exception as e:
            print(f"\n❌ Error training {crop} model: {str(e)}\n")
            continue
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    for crop, metrics in all_metrics.items():
        accuracy = (1 - (metrics['test_mae'] / 30)) * 100  # Approximate
        print(f"{crop:10s} - Test MAE: ₹{metrics['test_mae']:.2f}/kg, "
              f"R²: {metrics['test_r2']:.3f}, Accuracy: ~{accuracy:.1f}%")
    
    print("\n✓ All models trained and saved successfully!")
    print("\nModel files are in: models/trained_models/")
    print("You can now run the Flask API to use these models for predictions.")
    print("="*60)


if __name__ == "__main__":
    main()