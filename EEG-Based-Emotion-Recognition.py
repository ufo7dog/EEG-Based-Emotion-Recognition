import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import glob
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import pickle
import json
from datetime import datetime
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error


def load_all_subjects_data(brain_data_path, productivity_path):
    """Load and organize data from all subjects with improved error handling"""
    try:
        productivity_df = pd.read_excel(productivity_path)
        print(f"Loaded productivity data: {productivity_df.shape}")
    except Exception as e:
        print(f"Error loading productivity data: {e}")
        return None, None, None, None
    emotion_features = []
    productivity_scores = []
    brain_files = glob.glob(os.path.join(brain_data_path, "brain*.xlsx"))
    print(f"Found {len(brain_files)} brain data files")
    
    if not brain_files:
        print("No brain data files found! Check your path.")
        return None, None, None, None
    
    for brain_file in brain_files:
        try:
            subject_id = os.path.basename(brain_file).replace(".xlsx", "")
            print(f"Processing {subject_id}...")
            brain_df = pd.read_excel(brain_file)
            print(f"  Brain data shape: {brain_df.shape}")
            print(f"  Columns: {list(brain_df.columns)}")
            required_emotion_cols = ['Happy_valence_1', 'Happy_arousal_1', 'Sad_valence_1', 'Sad_arousal_1']
            required_prod_cols = ['HP_valence_1', 'HP_arousal_1', 'SD_valence_1', 'SD_arousal_1']
            missing_emotion_cols = [col for col in required_emotion_cols if col not in brain_df.columns]
            missing_prod_cols = [col for col in required_prod_cols if col not in brain_df.columns]
            if missing_emotion_cols:
                print(f"  Warning: Missing emotion columns: {missing_emotion_cols}")
            if missing_prod_cols:
                print(f"  Warning: Missing productivity columns: {missing_prod_cols}")
            if not missing_emotion_cols:
                happy_features = brain_df[['Happy_valence_1', 'Happy_arousal_1']].dropna().values
                sad_features = brain_df[['Sad_valence_1', 'Sad_arousal_1']].dropna().values
                print(f"  Happy features: {happy_features.shape}")
                print(f"  Sad features: {sad_features.shape}")
                for feature in happy_features:
                    emotion_features.append(np.append(feature, 1))  # 1 for happy
                
                for feature in sad_features:
                    emotion_features.append(np.append(feature, 0))  # 0 for sad
            if not missing_prod_cols:
                hp_features = brain_df[['HP_valence_1', 'HP_arousal_1']].dropna().values
                sd_features = brain_df[['SD_valence_1', 'SD_arousal_1']].dropna().values
                print(f"  HP features: {hp_features.shape}")
                print(f"  SD features: {sd_features.shape}")
                subject_productivity = productivity_df[productivity_df['Source'] == subject_id]
                if not subject_productivity.empty:
                    hp_productivity = subject_productivity['HP_productivity'].values
                    sd_productivity = subject_productivity['SD_productivity'].values
                    print(f"  HP productivity scores: {len(hp_productivity)}")
                    print(f"  SD productivity scores: {len(sd_productivity)}")
                    for i, feature in enumerate(hp_features):
                        if i < len(hp_productivity):
                            productivity_scores.append((feature, hp_productivity[i]))
                    
                    for i, feature in enumerate(sd_features):
                        if i < len(sd_productivity):
                            productivity_scores.append((feature, sd_productivity[i]))
                else:
                    print(f"  No productivity data found for {subject_id}")
        except Exception as e:
            print(f"Error processing {brain_file}: {e}")
            continue
    if not emotion_features:
        print("No emotion features found!")
        X_emotion, y_emotion = None, None
    else:
        emotion_features = np.array(emotion_features)
        X_emotion = emotion_features[:, :2]
        y_emotion = emotion_features[:, 2]
        print(f"Total emotion samples: {X_emotion.shape[0]}")
        print(f"Happy samples: {np.sum(y_emotion == 1)}")
        print(f"Sad samples: {np.sum(y_emotion == 0)}")
    
    if not productivity_scores:
        print("No productivity features found!")
        X_productivity, y_productivity = None, None
    else:
        X_productivity = np.array([item[0] for item in productivity_scores])
        y_productivity = np.array([item[1] for item in productivity_scores])
        print(f"Total productivity samples: {X_productivity.shape[0]}")
        print(f"Productivity score range: {y_productivity.min():.2f} - {y_productivity.max():.2f}")
    
    return X_emotion, y_emotion, X_productivity, y_productivity

def create_improved_emotion_classifier(input_dim=2):
    """Create a stable emotion classifier to avoid NaN losses"""
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.1),
        Dense(1, activation='sigmoid')])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model

def create_traditional_emotion_models():
    """Create SVM and Random Forest emotion classifiers"""
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    return svm_model, rf_model

def create_traditional_productivity_models():
    """Create SVM and Random Forest productivity predictors"""
    svm_model = SVR(kernel='rbf')
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    return svm_model, rf_model

def compare_model_performances(models_dict, X_test, y_test, model_type="classification"):
    """Compare performance of different models"""
    results = {}
    
    for name, model in models_dict.items():
        if model_type == "classification":
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:  # Neural network
                y_pred_proba = model.predict(X_test).flatten()
                y_pred = (y_pred_proba > 0.5).astype(int)
            
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {'accuracy': accuracy, 'predictions': y_pred}
            print(f"{name} Accuracy: {accuracy:.4f}")
        
        else:  # regression
            if hasattr(model, 'predict') and not hasattr(model, 'fit'):  # Neural network
                y_pred = model.predict(X_test).flatten()
            else:
                y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            results[name] = {'mae': mae, 'predictions': y_pred}
            print(f"{name} MAE: {mae:.4f}")
    
    return results

def create_improved_productivity_predictor(input_dim=2):
    """Create a better productivity predictor with proper scaling handling"""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.1),
        Dense(1, activation='linear')])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001, clipnorm=1.0),
        loss='mse',
        metrics=['mae', 'mse'])
    return model

def plot_training_history(history, title):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title(f'{title} - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    if 'accuracy' in history.history:
        axes[1].plot(history.history['accuracy'], label='Training Accuracy')
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_ylabel('Accuracy')
    else:
        axes[1].plot(history.history['mae'], label='Training MAE')
        axes[1].plot(history.history['val_mae'], label='Validation MAE')
        axes[1].set_ylabel('MAE')
    axes[1].set_title(f'{title} - Performance')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    plt.tight_layout()
    plt.show()

def plot_traditional_model_performance(results_dict, title, model_type="classification"):
    """Plot bar charts similar to training history for traditional models"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    names = list(results_dict.keys())

    if model_type == "classification":
        accuracies = [results_dict[name]['accuracy'] for name in names]
        axes[0].bar(names, accuracies, color='skyblue')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title(f'{title} - Accuracy Comparison')
        axes[0].set_ylim(0, 1)

        # dummy second plot to align with training history layout
        axes[1].axis('off')
        
    elif model_type == "regression":
        maes = [results_dict[name]['mae'] for name in names]
        axes[0].bar(names, maes, color='salmon')
        axes[0].set_ylabel('MAE')
        axes[0].set_title(f'{title} - MAE Comparison')

        # dummy second plot to align with training history layout
        axes[1].axis('off')
    
    axes[0].set_xlabel('Models')
    plt.tight_layout()
    plt.show()


def evaluate_emotion_model(model, X_test, y_test):
    """Comprehensive evaluation of emotion classification model"""
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Sad', 'Happy']))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Sad', 'Happy'], yticklabels=['Sad', 'Happy'])
    plt.title('Confusion Matrix - Emotion Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    return y_pred_proba, y_pred

def save_models_and_scalers(emotion_model, happy_productivity_model, sad_productivity_model, 
                          scaler_emotion, scaler_productivity, 
                          emotion_accuracy, happy_mae, sad_mae):
    """Save trained models, scalers, and performance metrics"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    emotion_model_path = f'emotion_classifier_{timestamp}.h5'
    happy_productivity_model_path = f'happy_productivity_predictor_{timestamp}.h5'
    sad_productivity_model_path = f'sad_productivity_predictor_{timestamp}.h5'
    
    emotion_model.save(emotion_model_path)
    
    # Only save productivity models if they exist
    if happy_productivity_model is not None:
        happy_productivity_model.save(happy_productivity_model_path)
    else:
        happy_productivity_model_path = None
        
    if sad_productivity_model is not None:
        sad_productivity_model.save(sad_productivity_model_path)
    else:
        sad_productivity_model_path = None
    
    # Save scalers...
    scaler_emotion_path = f'scaler_emotion_{timestamp}.pkl'
    scaler_productivity_path = f'scaler_productivity_{timestamp}.pkl'
    
    with open(scaler_emotion_path, 'wb') as f:
        pickle.dump(scaler_emotion, f)
    with open(scaler_productivity_path, 'wb') as f:
        pickle.dump(scaler_productivity, f)
    
    model_info = {
        'timestamp': timestamp,
        'emotion_model_path': emotion_model_path,
        'happy_productivity_model_path': happy_productivity_model_path,
        'sad_productivity_model_path': sad_productivity_model_path,
        'scaler_emotion_path': scaler_emotion_path,
        'scaler_productivity_path': scaler_productivity_path,
        'performance': {
            'emotion_accuracy': float(emotion_accuracy),
            'happy_productivity_mae': float(happy_mae) if happy_mae != float('inf') else None,
            'sad_productivity_mae': float(sad_mae) if sad_mae != float('inf') else None
        }
    }
    
    info_path = f'model_info_{timestamp}.json'
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    return model_info

def load_models_and_scalers(model_info_path):
    """Load trained models and scalers"""
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    emotion_model = tf.keras.models.load_model(model_info['emotion_model_path'])
    
    # Load productivity models if they exist
    happy_productivity_model = None
    sad_productivity_model = None
    
    if model_info['happy_productivity_model_path']:
        happy_productivity_model = tf.keras.models.load_model(model_info['happy_productivity_model_path'])
    
    if model_info['sad_productivity_model_path']:
        sad_productivity_model = tf.keras.models.load_model(model_info['sad_productivity_model_path'])
    
    with open(model_info['scaler_emotion_path'], 'rb') as f:
        scaler_emotion = pickle.load(f)
    with open(model_info['scaler_productivity_path'], 'rb') as f:
        scaler_productivity = pickle.load(f)
    
    print("Models loaded successfully!")
    print("Performance metrics:")
    print(f"  Emotion accuracy: {model_info['performance']['emotion_accuracy']:.4f}")
    if model_info['performance']['happy_productivity_mae']:
        print(f"  Happy Productivity MAE: {model_info['performance']['happy_productivity_mae']:.4f}")
    if model_info['performance']['sad_productivity_mae']:
        print(f"  Sad Productivity MAE: {model_info['performance']['sad_productivity_mae']:.4f}")
    
    return emotion_model, happy_productivity_model, sad_productivity_model, scaler_emotion, scaler_productivity, model_info
    
def analyze_data_distribution(X_emotion, y_emotion, X_productivity, y_productivity):
    """Analyze the distribution of your data"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    if X_emotion is not None:
        happy_data = X_emotion[y_emotion == 1]
        sad_data = X_emotion[y_emotion == 0]
        axes[0, 0].scatter(happy_data[:, 0], happy_data[:, 1], alpha=0.6, label='Happy', color='gold')
        axes[0, 0].scatter(sad_data[:, 0], sad_data[:, 1], alpha=0.6, label='Sad', color='blue')
        axes[0, 0].set_xlabel('Valence')
        axes[0, 0].set_ylabel('Arousal')
        axes[0, 0].set_title('Emotion Data Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].hist(happy_data[:, 0], alpha=0.7, label='Happy', bins=20, color='gold')
        axes[0, 1].hist(sad_data[:, 0], alpha=0.7, label='Sad', bins=20, color='blue')
        axes[0, 1].set_xlabel('Valence')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Valence Distribution by Emotion')
        axes[0, 1].legend()
    
    if X_productivity is not None:
        scatter = axes[1, 0].scatter(X_productivity[:, 0], X_productivity[:, 1], 
                                   c=y_productivity, cmap='viridis', alpha=0.6)
        axes[1, 0].set_xlabel('Valence')
        axes[1, 0].set_ylabel('Arousal')
        axes[1, 0].set_title('Productivity Data Distribution')
        plt.colorbar(scatter, ax=axes[1, 0], label='Productivity Score')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].hist(y_productivity, bins=20, alpha=0.7, color='green')
        axes[1, 1].set_xlabel('Productivity Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Productivity Score Distribution')
        axes[1, 1].axvline(y_productivity.mean(), color='red', linestyle='--', 
                          label=f'Mean: {y_productivity.mean():.2f}')
        axes[1, 1].legend()
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function with improved error handling"""
    np.random.seed(42)
    tf.random.set_seed(42)
    print("=== EEG Emotion & Productivity Analysis ===")
    
    # Initialize model variables at the beginning
    emotion_model = None
    happy_productivity_model = None
    sad_productivity_model = None
    scaler_emotion = None
    scaler_productivity = None
    emotion_accuracy = 0.0
    happy_mae = float('inf')
    sad_mae = float('inf')
    
    brain_data_path = './v3_cleaned_data/Outputs/'
    productivity_path = 'v3_cleaned_data/productivity_data.xlsx'
    print("Loading data from:")
    print(f"  Brain data: {brain_data_path}")
    print(f"  Productivity data: {productivity_path}")
    X_emotion, y_emotion, X_productivity, y_productivity = load_all_subjects_data(
        brain_data_path, productivity_path)
    
    if X_emotion is None and X_productivity is None:
        print("No data loaded successfully. Please check your file paths and data format.")
        return
    
    if X_emotion is not None:
        print("\nEmotion data quality check:")
        print(f"  Data shape: {X_emotion.shape}")
        print(f"  Contains NaN: {np.isnan(X_emotion).any()}")
        print(f"  Contains Inf: {np.isinf(X_emotion).any()}")
        print(f"  Data range: [{X_emotion.min():.4f}, {X_emotion.max():.4f}]")
        valid_mask = ~(np.isnan(X_emotion).any(axis=1) | np.isinf(X_emotion).any(axis=1))
        X_emotion = X_emotion[valid_mask]
        y_emotion = y_emotion[valid_mask]
        print(f"  After cleaning: {X_emotion.shape}")
    
    if X_productivity is not None:
        print("\nProductivity data quality check:")
        print(f"  Data shape: {X_productivity.shape}")
        print(f"  Contains NaN: {np.isnan(X_productivity).any() or np.isnan(y_productivity).any()}")
        print(f"  Contains Inf: {np.isinf(X_productivity).any() or np.isinf(y_productivity).any()}")
        print(f"  Feature range: [{X_productivity.min():.4f}, {X_productivity.max():.4f}]")
        print(f"  Target range: [{y_productivity.min():.4f}, {y_productivity.max():.4f}]")
        valid_mask = ~(np.isnan(X_productivity).any(axis=1) | np.isinf(X_productivity).any(axis=1) | 
                      np.isnan(y_productivity) | np.isinf(y_productivity))
        X_productivity = X_productivity[valid_mask]
        y_productivity = y_productivity[valid_mask]
        print(f"  After cleaning: {X_productivity.shape}")
    
    print("\nAnalyzing data distribution...")
    # analyze_data_distribution(X_emotion, y_emotion, X_productivity, y_productivity)
    
    # Train emotion classifier
    if X_emotion is not None:
        print("\n=== Training Emotion Classifier ===")
        scaler_emotion = StandardScaler()
        X_emotion_scaled = scaler_emotion.fit_transform(X_emotion)
        X_train, X_test, y_train, y_test = train_test_split(
            X_emotion_scaled, y_emotion, test_size=0.2, random_state=42, stratify=y_emotion)
        emotion_model = create_improved_emotion_classifier()
        callbacks = [EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, min_delta=1e-6),]
        emotion_history = emotion_model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=min(32, len(X_train)//4),
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1)
        emotion_loss, emotion_accuracy = emotion_model.evaluate(X_test, y_test, verbose=0)[:2]
        
        svm_emotion, rf_emotion = create_traditional_emotion_models()
        svm_emotion.fit(X_train, y_train)
        rf_emotion.fit(X_train, y_train)
        emotion_models = {
            'Neural Network': emotion_model,
            'SVM': svm_emotion,
            'Random Forest': rf_emotion}
        
        print("\nEmotion Classification Results:")
        print(f"  Test Accuracy: {emotion_accuracy:.4f}")
        print(f"  Test Loss: {emotion_loss:.4f}")
        y_pred_proba, y_pred = evaluate_emotion_model(emotion_model, X_test, y_test)        
        plot_training_history(emotion_history, "Emotion Classification")
        
        emotion_results = compare_model_performances(emotion_models, X_test, y_test, "classification")
        plot_traditional_model_performance(emotion_results, "Emotion Classification", model_type="classification")

        # Use best model for productivity prediction
        best_emotion_model = max(emotion_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"Best emotion model: {best_emotion_model[0]} (Accuracy: {best_emotion_model[1]['accuracy']:.4f})")

        
    # Train productivity predictors
    if X_productivity is not None and emotion_model is not None:
        print("\n=== Training Separate Productivity Predictors ===")
        scaler_productivity = StandardScaler()
        X_productivity_scaled = scaler_productivity.fit_transform(X_productivity)
        
        best_model_name, best_model_data = best_emotion_model
        best_model = emotion_models[best_model_name]
        if hasattr(best_model, 'predict_proba'):
            emotion_pred_for_prod = best_model.predict_proba(X_productivity_scaled)[:, 1]
        else:
            emotion_pred_for_prod = best_model.predict(X_productivity_scaled).flatten()
        
        
        # First predict emotions for productivity data
        # emotion_pred_for_prod = emotion_model.predict(X_productivity_scaled)
        predicted_emotions = (emotion_pred_for_prod > 0.5).astype(int).flatten()
        
        # Split data by predicted emotion
        happy_mask = predicted_emotions == 1
        sad_mask = predicted_emotions == 0
        
        X_happy = X_productivity_scaled[happy_mask]
        y_happy = y_productivity[happy_mask]
        X_sad = X_productivity_scaled[sad_mask]
        y_sad = y_productivity[sad_mask]
        
        print(f"Happy productivity samples: {len(X_happy)}")
        print(f"Sad productivity samples: {len(X_sad)}")
        
        # Train happy productivity model
        if len(X_happy) > 10:
            print("\nTraining Happy Productivity Models...")
            X_train_happy, X_test_happy, y_train_happy, y_test_happy = train_test_split(
                X_happy, y_happy, test_size=0.2, random_state=42)
            
            # Neural Network
            happy_productivity_model = create_improved_productivity_predictor()
            happy_productivity_model.fit(X_train_happy, y_train_happy, epochs=150, 
                                       batch_size=min(16, len(X_train_happy)//4), 
                                       validation_split=0.2, verbose=0)
            
            # Traditional models
            svm_happy, rf_happy = create_traditional_productivity_models()
            svm_happy.fit(X_train_happy, y_train_happy)
            rf_happy.fit(X_train_happy, y_train_happy)
            
            happy_models = {
                'Neural Network': happy_productivity_model,
                'SVM': svm_happy,
                'Random Forest': rf_happy
            }
            
            print("Happy Productivity Results:")
            happy_results = compare_model_performances(happy_models, X_test_happy, y_test_happy, "regression")
            plot_traditional_model_performance(happy_results, "Happy Productivity", model_type="regression")
            
        
        # Train sad productivity model
        if len(X_sad) > 10:
            print("\nTraining Sad Productivity Models...")
            X_train_sad, X_test_sad, y_train_sad, y_test_sad = train_test_split(
                X_sad, y_sad, test_size=0.2, random_state=42)
            
            # Neural Network
            sad_productivity_model = create_improved_productivity_predictor()
            sad_productivity_model.fit(X_train_sad, y_train_sad, epochs=150,
                                     batch_size=min(16, len(X_train_sad)//4),
                                     validation_split=0.2, verbose=0)
            
            # Traditional models
            svm_sad, rf_sad = create_traditional_productivity_models()
            svm_sad.fit(X_train_sad, y_train_sad)
            rf_sad.fit(X_train_sad, y_train_sad)
            
            sad_models = {
                'Neural Network': sad_productivity_model,
                'SVM': svm_sad,
                'Random Forest': rf_sad
            }
            
            print("Sad Productivity Results:")
            sad_results = compare_model_performances(sad_models, X_test_sad, y_test_sad, "regression")
            plot_traditional_model_performance(sad_results, "Sad Productivity", model_type="regression")
            
    best_happy = min(happy_results.items(), key=lambda x: x[1]['mae']) if len(X_happy) > 10 else None
    best_sad = min(sad_results.items(), key=lambda x: x[1]['mae']) if len(X_sad) > 10 else None
    print("\n=== SUMMARY ===")
    print(f"Best emotion model: {best_emotion_model[0]} (Accuracy: {best_emotion_model[1]['accuracy']:.4f})")
    if best_happy:
        print(f"Best happy productivity model: {best_happy[0]} (MAE: {best_happy[1]['mae']:.4f})")
    if best_sad:
        print(f"Best sad productivity model: {best_sad[0]} (MAE: {best_sad[1]['mae']:.4f})")

    print("\n=== Analysis Complete ===")
    
    # Save models only if they exist
    if emotion_model is not None:
        model_info = save_models_and_scalers(
            emotion_model, happy_productivity_model, sad_productivity_model,
            scaler_emotion, scaler_productivity,
            emotion_accuracy, happy_mae, sad_mae)
        print(f"Models saved with timestamp: {model_info['timestamp']}")

    return emotion_model, happy_productivity_model, sad_productivity_model, scaler_emotion, scaler_productivity

if __name__ == "__main__":
    main()