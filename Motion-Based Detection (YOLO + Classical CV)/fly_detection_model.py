import os
import numpy as np
import pandas as pd
import pickle
import librosa
import json
from datetime import datetime
from scipy.signal import butter, sosfilt, filtfilt, savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc,
    precision_recall_curve,
    average_precision_score
)
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed


class FlyDetectionModel:
    def __init__(self, config=None):
        # Load configuration or use defaults
        self.config = config or {
            'sample_rate': 44100,
            'segment_length': 2.0,
            'lowcut': 50,
            'highcut': 150,
            'n_estimators': 100,
            'n_jobs': -1,
            'random_state': 42,
            'n_folds': 5
        }
        
        # Model components
        self.scaler = None
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.training_history = []

    def extract_features(self, audio_segment):
        try:
            print(f"\nProcessing steps:")
            print(f"1. Initial segment length: {len(audio_segment)}")
            
            # Preprocessing
            if len(audio_segment.shape) == 2:  # Stereo audio
                print("Converting stereo to mono")
                audio_segment = np.mean(audio_segment, axis=1)
            
            segment = audio_segment - np.mean(audio_segment)
            print(f"2. After mean subtraction length: {len(segment)}")
            
            # Bandpass filter
            nyq = 0.5 * self.config['sample_rate']
            lowcut_normalized = self.config['lowcut'] / nyq
            highcut_normalized = self.config['highcut'] / nyq
            print(f"3. Filter parameters:")
            print(f"   - Nyquist frequency: {nyq}")
            print(f"   - Normalized frequencies: {lowcut_normalized}, {highcut_normalized}")
            
            try:
                b_fly, a_fly = butter(3, [lowcut_normalized, highcut_normalized], btype='band')
                segment = filtfilt(b_fly, a_fly, segment, padtype='odd', padlen=None)
                print("4. Used filtfilt successfully")
            except ValueError as ve:
                print("4. filtfilt failed, trying sosfilt")
                sos = butter(3, [lowcut_normalized, highcut_normalized], btype='band', output='sos')
                segment = sosfilt(sos, segment)
            
            print(f"5. After filter length: {len(segment)}")
            
            # Savitzky-Golay filter
            sg_window = int(self.config['sample_rate'] / 50)
            sg_window = min(len(segment) - 1, sg_window)  # Ensure sg_window <= len(segment)
            if sg_window % 2 == 0:
                sg_window -= 1  # Ensure sg_window is odd
            
            print(f"6. SG window parameters:")
            print(f"   - Window size: {sg_window}")
            print(f"   - Segment length: {len(segment)}")
            
            segment = savgol_filter(segment, sg_window, 3, mode='mirror')
            print(f"7. After savgol length: {len(segment)}")
            
            # Normalize
            if len(segment.shape) == 1:  # Ensure segment is 1D
                segment = segment / np.max(np.abs(segment))
            else:
                raise ValueError(f"Unexpected segment shape: {segment.shape}")
            
            # Extract features
            n_fft = min(len(segment), 2048)  # Dynamically adjust n_fft
            mfcc = librosa.feature.mfcc(y=segment, sr=self.config['sample_rate'], n_mfcc=13, n_fft=n_fft)
            rms = librosa.feature.rms(y=segment)
            zcr = librosa.feature.zero_crossing_rate(y=segment)
            spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=self.config['sample_rate'], n_fft=n_fft)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=self.config['sample_rate'], n_fft=n_fft)
            
            features = {
                "MFCC_Mean": np.mean(mfcc, axis=1),
                "RMS": np.mean(rms),
                "Zero_Crossing_Rate": np.mean(zcr),
                "Spectral_Centroid": np.mean(spectral_centroid),
                "Spectral_Bandwidth": np.mean(spectral_bandwidth)
            }
            
            return features, None
        
        except Exception as e:
            import traceback
            print(f"Full error: {traceback.format_exc()}")
            return None, str(e)

    def _prepare_feature_vector(self, features_dict):
        """Convert features dictionary to vector format."""
        feature_vector = []
        
        # Add MFCC features
        feature_vector.extend(features_dict["MFCC_Mean"])
        
        # Add other features
        feature_vector.extend([
            features_dict["RMS"],
            features_dict["Zero_Crossing_Rate"],
            features_dict["Spectral_Centroid"],
            features_dict["Spectral_Bandwidth"]
        ])
        
        return np.array(feature_vector).reshape(1, -1)

    def perform_cross_validation(self, X, y):
        """Perform k-fold cross-validation."""
        kf = KFold(n_splits=self.config['n_folds'], shuffle=True, 
                  random_state=self.config['random_state'])
        
        cv_scores = cross_val_score(self.model, X, y, cv=kf, scoring='accuracy')
        
        return {
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }

    def plot_learning_curves(self, X_train, X_test, y_train, y_test):
        """Plot learning curves for model performance analysis."""
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_scores = []
        test_scores = []
        
        for size in train_sizes:
            n_samples = int(len(X_train) * size)
            model = RandomForestClassifier(**self.config)
            model.fit(X_train[:n_samples], y_train[:n_samples])
            
            train_scores.append(model.score(X_train[:n_samples], y_train[:n_samples]))
            test_scores.append(model.score(X_test, y_test))
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores, label='Training score')
        plt.plot(train_sizes, test_scores, label='Testing score')
        plt.xlabel('Training set size ratio')
        plt.ylabel('Score')
        plt.title('Learning Curves')
        plt.legend()
        
        return plt.gcf()

    def train(self, features_csv, model_dir='models', plot_curves=True):
        """Train the model using features from CSV with enhanced evaluation and MLflow tracking"""
        with mlflow.start_run() as run:
            print("Loading training data...")
            df = pd.read_csv(features_csv, sep='\t')
            
            # Log parameters
            mlflow.log_params(self.config)
            
            # Check columns and print available ones
            print("Available columns:", df.columns.tolist())
            
            # Prepare features based on available columns
            feature_cols = []
            X = pd.DataFrame()
            
            # Handle MFCC features if present
            if 'mfcc_mean' in df.columns:
                mfcc_col = 'mfcc_mean'
            elif 'MFCC_Mean' in df.columns:
                mfcc_col = 'MFCC_Mean'
            
            if mfcc_col in df.columns:
                try:
                    # Try to convert string representation of array to actual array
                    df[mfcc_col] = df[mfcc_col].apply(lambda x: eval(x) if isinstance(x, str) else x)
                    mfcc_df = pd.DataFrame(df[mfcc_col].tolist(), 
                                         columns=[f'MFCC_{i}' for i in range(13)])
                    X = pd.concat([X, mfcc_df], axis=1)
                except Exception as e:
                    print(f"Error processing MFCC features: {e}")
                    print("Skipping MFCC features")
            
            # Add other acoustic features if present
            potential_features = ['RMS', 'Zero_Crossing_Rate', 
                                'Spectral_Centroid', 'Spectral_Bandwidth',
                                'rms', 'zero_crossing_rate', 
                                'spectral_centroid', 'spectral_bandwidth']
            
            for feature in potential_features:
                if feature in df.columns:
                    X[feature] = df[feature]
                    feature_cols.append(feature)
            
            if X.empty:
                raise ValueError("No valid features found in CSV")
            
            self.feature_names = X.columns.tolist()
            
            # Create labels - check for different possible label column names
            label_cols = ['label', 'class', 'Segment']
            label_col = next((col for col in label_cols if col in df.columns), None)
            
            if not label_col:
                raise ValueError("No label column found in CSV")
            
            y = df[label_col].str.startswith(('fly', 'Fly')).astype(int)
            
            # Continue with model training as before...
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, 
                random_state=self.config['random_state'], 
                stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            print("Training Random Forest model...")
            self.model = RandomForestClassifier(
                n_estimators=self.config['n_estimators'],
                n_jobs=self.config['n_jobs'],
                random_state=self.config['random_state']
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Rest of the evaluation code remains the same...
            self.is_trained = True
            return run.info.run_id

    def plot_evaluation_curves(self, X_test_scaled, y_test, y_pred_proba, 
                             fpr, tpr, precision, recall, model_dir):
        """Plot and save evaluation curves."""
        # Create directory for plots
        plots_dir = os.path.join(model_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # ROC Curve
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.savefig(os.path.join(plots_dir, 'roc_curve.png'))
        plt.close()
        
        # Precision-Recall Curve
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig(os.path.join(plots_dir, 'precision_recall_curve.png'))
        plt.close()
        
        # Feature Importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
        plt.close()

    def batch_predict(self, audio_files, batch_size=32, n_workers=4):
        """Perform batch predictions on multiple audio files."""
        if not self.is_trained:
            raise ValueError("Model not trained or loaded!")
        
        results = {}
        
        def process_file(audio_path):
            try:
                result = self.predict_audio_file(audio_path)
                return audio_path, result
            except Exception as e:
                return audio_path, {'error': str(e)}
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(process_file, file) for file in audio_files]
            
            for future in tqdm(as_completed(futures), 
                             total=len(futures), 
                             desc="Processing audio files"):
                file_path, result = future.result()
                results[file_path] = result
        
        return results

    def predict_audio_file(self, audio_path):
    #"""Predict on a new audio file with enhanced error handling."""
        if not self.is_trained:
            raise ValueError("Model not trained or loaded!")
            
        try:
            # Load audio
            audio_data, sr = sf.read(audio_path)
            
            # Ensure correct sample rate
            if sr != self.config['sample_rate']:
                audio_data = librosa.resample(audio_data, 
                                        orig_sr=sr, 
                                        target_sr=self.config['sample_rate'])
            
            # Extract features
            features, error = self.extract_features(audio_data)
            if error:
                return {'error': f"Feature extraction failed: {error}"}
                
            feature_vector = self._prepare_feature_vector(features)
            
            # Scale features
            scaled_features = self.scaler.transform(feature_vector)
            
            # Predict
            prediction = self.model.predict(scaled_features)[0]
            probabilities = self.model.predict_proba(scaled_features)[0]
            
            return {
                'prediction': 'fly' if prediction == 1 else 'no_fly',
                'confidence': float(np.max(probabilities)),
                'probabilities': {
                    'no_fly': float(probabilities[0]),
                    'fly': float(probabilities[1])
                },
                'processing_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': f"Prediction failed: {str(e)}"}
    
    def predict_long_audio(self, audio_path, segment_length=2.0):
        """Predict on a longer audio file by analyzing segments."""
        if not self.is_trained:
            raise ValueError("Model not trained or loaded!")
            
        try:
            # Load audio
            print(f"Loading audio file: {audio_path}")
            audio_data, sr = sf.read(audio_path)
            print(f"Audio length: {len(audio_data)} samples ({len(audio_data)/sr:.2f} seconds)")
            
            # Ensure correct sample rate
            if sr != self.config['sample_rate']:
                print(f"Resampling from {sr} to {self.config['sample_rate']}")
                audio_data = librosa.resample(audio_data, 
                                            orig_sr=sr, 
                                            target_sr=self.config['sample_rate'])
            
            # Calculate segments
            samples_per_segment = int(segment_length * self.config['sample_rate'])
            total_segments = len(audio_data) // samples_per_segment
            print(f"Will process {total_segments} segments of {segment_length} seconds each")
            
            predictions = []
            valid_predictions = 0
            
            # Process each segment
            for i in range(total_segments):
                # Extract segment
                start = i * samples_per_segment
                end = start + samples_per_segment
                segment = audio_data[start:end]
                
                # Extract features for segment
                features, error = self.extract_features(segment)
                if error:
                    print(f"Skipping segment {i}: {error}")
                    continue
                
                if features is None:
                    continue
                    
                # Prepare for prediction
                feature_vector = self._prepare_feature_vector(features)
                scaled_features = self.scaler.transform(feature_vector)
                
                # Make prediction
                prediction = self.model.predict(scaled_features)[0]
                probabilities = self.model.predict_proba(scaled_features)[0]
                
                # Store results
                predictions.append({
                    'segment': i,
                    'time_start': start / self.config['sample_rate'],
                    'time_end': end / self.config['sample_rate'],
                    'prediction': 'fly' if prediction == 1 else 'no_fly',
                    'confidence': float(np.max(probabilities)),
                    'probabilities': {
                        'no_fly': float(probabilities[0]),
                        'fly': float(probabilities[1])
                    }
                })
                valid_predictions += 1
                
                # Print progress
                if (i + 1) % 5 == 0:
                    print(f"Processed {i + 1}/{total_segments} segments")
            
            if valid_predictions == 0:
                return {
                    'error': 'No valid segments could be processed',
                    'total_segments': total_segments,
                    'valid_segments': 0
                }
            
            # Compute summary statistics
            fly_segments = sum(1 for p in predictions if p['prediction'] == 'fly')
            no_fly_segments = valid_predictions - fly_segments
            
            # Create time series of predictions
            time_series = [{
                'time': p['time_start'],
                'prediction': p['prediction'],
                'confidence': p['confidence']
            } for p in predictions]
            
            # Summarize results
            summary = {
                'total_segments': total_segments,
                'valid_segments': valid_predictions,
                'fly_segments': fly_segments,
                'no_fly_segments': no_fly_segments,
                'average_confidence': np.mean([p['confidence'] for p in predictions]),
                'detection_rate': fly_segments / valid_predictions if valid_predictions > 0 else 0,
                'segment_predictions': predictions,
                'time_series': time_series,
                'audio_duration': len(audio_data) / self.config['sample_rate'],
                'processing_time': datetime.now().isoformat()
            }
            
            print("\nAnalysis Complete:")
            print(f"Processed {valid_predictions} valid segments")
            print(f"Found {fly_segments} segments with fly activity")
            print(f"Average confidence: {summary['average_confidence']:.2%}")
            
            return summary
            
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return {'error': f"Prediction failed: {str(e)}"}

    def load_model(self, model_path):
        """Load a trained model from file with verification."""
        try:
            with open(model_path, 'rb') as f:
                components = pickle.load(f)
                self.model = components['model']
                self.scaler = components['scaler']
                self.config = components.get('config', self.config)
                self.feature_names = components.get('feature_names')
            
            self.is_trained = True
            print("Model loaded successfully!")
            
            # Verify model components
            assert hasattr(self.model, 'predict'), "Invalid model object"
            assert hasattr(self.scaler, 'transform'), "Invalid scaler object"
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.is_trained = False
            return False

    def batch_predict(self, audio_files, batch_size=32, n_workers=4):
        """Perform batch predictions on multiple audio files."""
        if not self.is_trained:
            raise ValueError("Model not trained or loaded!")
        
        results = {}
        
        def process_file(audio_path):
            try:
                result = self.predict_audio_file(audio_path)
                return audio_path, result
            except Exception as e:
                return audio_path, {'error': str(e)}
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(process_file, file) for file in audio_files]
            
            for future in tqdm(as_completed(futures), 
                            total=len(futures), 
                            desc="Processing audio files"):
                file_path, result = future.result()
                results[file_path] = result
        
        return results

# Example usage (OLD)
# if __name__ == "__main__":
#     # Initialize model with custom configuration
#     config = {
#         'sample_rate': 44100,
#         'segment_length': 2.0,
#         'lowcut': 50,
#         'highcut': 150,
#         'n_estimators': 100,
#         'n_jobs': -1,
#         'random_state': 42,
#         'n_folds': 5
#     }
    
#     detector = FlyDetectionModel(config)
    
#     # Train or load model
#     if os.path.exists('models/fly_detector.pkl'):
#         detector.load_model('models/fly_detector.pkl')
#     else:
#         # Train new model
#         results = detector.train('features.csv')
#         print("\nTraining Results:")
#         print(f"Cross-validation score: {results['cross_validation']['mean_cv_score']:.3f}")
#         print(f"ROC AUC: {results['roc_auc']:.3f}")
#         print(f"Average Precision: {results['avg_precision']:.3f}")
    
#     # Example batch prediction
#     test_files = ['test1.wav', 'test2.wav', 'test3.wav']
#     if all(os.path.exists(f) for f in test_files):
#         results = detector.batch_predict(test_files)
#         print("\nBatch Prediction Results:")
#         for file, result in results.items():
#             print(f"\nFile: {file}")
#             if 'error' in result:
#                 print(f"Error: {result['error']}")
#             else:
#                 print(f"Prediction: {result['prediction']}")
#                 print(f"Confidence: {result['confidence']:.2%}")

if __name__ == "__main__":
    # Set up MLflow tracking URI and experiment
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fly_detection")

    # Initialize model with custom configuration
    config = {
        'sample_rate': 44100,
        'segment_length': 2.0,
        'lowcut': 50,
        'highcut': 150,
        'n_estimators': 100,
        'n_jobs': -1,
        'random_state': 42,
        'n_folds': 5
    }
    
    detector = FlyDetectionModel(config)
    
    # Define paths
    data_dir = "/home/pc/LG/PSU/yolo_dino/src/data/gdrive/cantor_data"  # Make sure this directory exists
    model_dir = "/home/pc/LG/PSU/yolo_dino/src/models"  # Make sure this directory exists
    features_path = os.path.join(data_dir, "features.csv")
    model_path = os.path.join(model_dir, "fly_detector.pkl")
    
    # Create necessary directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Check if features file exists
    if not os.path.exists(features_path):
        print(f"Error: Features file not found at {features_path}")
        print("Please ensure your features.csv file is in the data directory")
        exit(1)
    
    # Train or load model
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        detector.load_model(model_path)
    else:
        print(f"Training new model using features from {features_path}")
        run_id = detector.train(features_path)
        print(f"\nTraining completed. MLflow run ID: {run_id}")
        print("\nView results in MLflow UI by running:")
        print("mlflow ui")
    
    # Example batch prediction
    test_dir = os.path.join(data_dir, "test")
    test_files = [os.path.join(test_dir, f) for f in ['test1.wav', 'test2.wav', 'test3.wav']]
    
    if all(os.path.exists(f) for f in test_files):
        results = detector.batch_predict(test_files)
        print("\nBatch Prediction Results:")
        for file, result in results.items():
            print(f"\nFile: {file}")
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.2%}")
    else:
        print("\nTest files not found. Skipping prediction.")