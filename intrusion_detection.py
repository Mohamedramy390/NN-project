# =============================================================================
# Cybersecurity Intrusion Detection System
# IMPROVED SUPERVISED LEARNING VERSION
# =============================================================================

import os
import warnings
# Configure environment first
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_DISABLE_PROGRESS_BAR'] = '1'
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, precision_recall_curve, auc,
                           accuracy_score, precision_recall_fscore_support)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, 
    BatchNormalization, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("🚀 Libraries imported successfully!")
print(f"🧠 TensorFlow version: {tf.__version__}")

# =============================================================================
# DATA LOADING AND EXPLORATION
# =============================================================================

def load_data():
    """
    Load the pre-split UNSW-NB15 training and testing sets
    """
    try:
        print("📥 Loading UNSW-NB15 dataset...")
        
        # Load the pre-split files
        train_df = pd.read_csv("UNSW_NB15_training-set.csv")
        test_df = pd.read_csv("UNSW_NB15_testing-set.csv")
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   Training set: {train_df.shape}")
        print(f"   Testing set: {test_df.shape}")
        
        # Display dataset info
        print(f"\n📊 Training set label distribution:")
        print(train_df['label'].value_counts())
        
        if 'attack_cat' in train_df.columns:
            print(f"\n🎯 Attack categories in training set:")
            print(train_df['attack_cat'].value_counts())
        
        return train_df, test_df
        
    except FileNotFoundError as e:
        print(f"❌ Error loading files: {e}")
        print("⚠ Creating synthetic data for demonstration...")
        return create_synthetic_data()

def create_synthetic_data():
    """Create synthetic network traffic data for demonstration"""
    np.random.seed(42)
    n_train = 10000
    n_test = 2000
    
    # Feature names based on UNSW-NB15 dataset characteristics
    features = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
        'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
        'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
    ]
    
    # Create synthetic training data
    train_data = {}
    for feature in features:
        if 'protocol' in feature or 'service' in feature or 'flag' in feature:
            train_data[feature] = np.random.randint(0, 10, n_train)
        else:
            train_data[feature] = np.random.exponential(1, n_train)
    
    train_df = pd.DataFrame(train_data)
    
    # Create labels (90% normal, 10% attacks)
    train_labels = np.random.choice([0, 1], size=n_train, p=[0.9, 0.1])
    train_df['label'] = train_labels
    
    # Create attack categories
    attack_categories = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R', 'Generic']
    attack_cats = ['Normal'] * n_train
    for i in range(n_train):
        if train_labels[i] == 1:
            attack_cats[i] = np.random.choice(attack_categories[1:])
    train_df['attack_cat'] = attack_cats
    
    # Create testing data
    test_data = {}
    for feature in features:
        if 'protocol' in feature or 'service' in feature or 'flag' in feature:
            test_data[feature] = np.random.randint(0, 10, n_test)
        else:
            test_data[feature] = np.random.exponential(1, n_test)
    
    test_df = pd.DataFrame(test_data)
    test_labels = np.random.choice([0, 1], size=n_test, p=[0.85, 0.15])
    test_df['label'] = test_labels
    
    attack_cats_test = ['Normal'] * n_test
    for i in range(n_test):
        if test_labels[i] == 1:
            attack_cats_test[i] = np.random.choice(attack_categories[1:])
    test_df['attack_cat'] = attack_cats_test
    
    print("✓ Synthetic data created successfully!")
    return train_df, test_df

# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def preprocess_data(train_df, test_df):
    """
    Simplified preprocessing that handles categorical variables safely
    """
    print("\n🔄 Preprocessing data...")
    
    # Make copies
    train_df_processed = train_df.copy()
    test_df_processed = test_df.copy()
    
    # Remove unnecessary columns
    columns_to_drop = ['id'] if 'id' in train_df.columns else []
    for col in columns_to_drop:
        train_df_processed = train_df_processed.drop(columns=[col], errors='ignore')
        test_df_processed = test_df_processed.drop(columns=[col], errors='ignore')
    
    # Handle categorical columns - use one-hot encoding
    categorical_columns = []
    for col in train_df_processed.columns:
        if train_df_processed[col].dtype == 'object' and col not in ['label', 'attack_cat']:
            categorical_columns.append(col)
    
    print(f"🔧 One-hot encoding {len(categorical_columns)} categorical columns...")
    
    # Combine train and test for one-hot encoding to ensure same columns
    combined_df = pd.concat([train_df_processed, test_df_processed], ignore_index=True)
    
    # One-hot encode categorical variables
    combined_encoded = pd.get_dummies(combined_df, columns=categorical_columns, drop_first=True)
    
    # Split back to train and test
    train_size = len(train_df_processed)
    train_encoded = combined_encoded.iloc[:train_size]
    test_encoded = combined_encoded.iloc[train_size:]
    
    # Separate features and labels
    feature_columns = [col for col in train_encoded.columns if col not in ['label', 'attack_cat']]
    
    X_train = train_encoded[feature_columns]
    X_test = test_encoded[feature_columns]
    y_train = train_encoded['label']
    y_test = test_encoded['label']
    
    # Get attack categories
    attack_cat_train = train_encoded.get('attack_cat', None)
    attack_cat_test = test_encoded.get('attack_cat', None)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Preprocessing completed!")
    print(f"  Training features: {X_train_scaled.shape}")
    print(f"  Testing features: {X_test_scaled.shape}")
    print(f"  Total features after encoding: {len(feature_columns)}")
    
    return (X_train_scaled, X_test_scaled, y_train, y_test, 
            attack_cat_train, attack_cat_test, scaler, {}, feature_columns)

# =============================================================================
# SEQUENCE CREATION
# =============================================================================

def create_sequences(X, y, sequence_length=20, step_size=2):
    """
    Convert tabular data into sequences for temporal modeling
    """
    sequences = []
    sequence_labels = []
    
    n_samples = len(X)
    
    for i in range(0, n_samples - sequence_length, step_size):
        sequence = X[i:i + sequence_length]
        label = y.iloc[i + sequence_length - 1] if hasattr(y, 'iloc') else y[i + sequence_length - 1]
        
        sequences.append(sequence)
        sequence_labels.append(label)
    
    # Convert to float32 to save memory
    sequences_array = np.array(sequences, dtype=np.float32)
    labels_array = np.array(sequence_labels)
    
    print(f"  Created {len(sequences_array)} sequences of length {sequence_length}")
    print(f"  Normal sequences: {np.sum(labels_array == 0)}")
    print(f"  Attack sequences: {np.sum(labels_array == 1)}")
    
    return sequences_array, labels_array

def prepare_sequences(X_train, X_test, y_train, y_test, sequence_length=20):
    """
    Prepare sequences for training and testing
    """
    print(f"\n📦 Creating sequences (length: {sequence_length})...")
    
    # Create sequences for training and testing
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length, step_size=2)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length, step_size=2)
    
    print(f"✓ Sequence creation completed!")
    print(f"  Training sequences: {X_train_seq.shape}")
    print(f"  Testing sequences: {X_test_seq.shape}")
    
    return X_train_seq, X_test_seq, y_train_seq, y_test_seq

# =============================================================================
# MODEL ARCHITECTURE - SUPERVISED LEARNING
# =============================================================================

def build_improved_model(sequence_length, n_features):
    """
    Build a BETTER model for intrusion detection
    Uses supervised learning instead of autoencoder
    """
    print(f"\n🏗️ Building Improved CNN-LSTM Classifier...")
    
    inputs = Input(shape=(sequence_length, n_features))
    
    # CNN for feature extraction
    x = Conv1D(128, 3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    
    # LSTM for temporal patterns
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(32, return_sequences=False)(x)
    x = Dropout(0.3)(x)
    
    # Classification head
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)  # Binary classification
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("✓ Improved model built successfully!")
    return model

# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_improved_model(model, X_train_seq, y_train_seq, X_val_seq, y_val_seq):
    """
    Train the improved model
    """
    print("\n🎯 Training Improved Model...")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]
    
    # Use class weights for imbalanced data
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_seq),
        y=y_train_seq
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    print("✓ Training completed!")
    return history

# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_improved_model(model, X_test_seq, y_test_seq):
    """
    Comprehensive evaluation
    """
    print("\n📊 Evaluating Improved Model...")
    
    # Predictions
    y_pred_proba = model.predict(X_test_seq, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_seq, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_seq, y_pred, average='binary', zero_division=0
    )
    auc_roc = roc_auc_score(y_test_seq, y_pred_proba)
    
    # AUC-PR
    precision_curve, recall_curve, _ = precision_recall_curve(y_test_seq, y_pred_proba)
    auc_pr = auc(recall_curve, precision_curve)
    
    print("🎯 IMPROVED PERFORMANCE METRICS:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc_roc:.4f}")
    print(f"  AUC-PR:    {auc_pr:.4f}")
    
    # Detailed report
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y_test_seq, y_pred, target_names=['Normal', 'Attack']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'],
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss During Training', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy During Training', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Accuracy', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("🔒 IMPROVED CYBERSECURITY INTRUSION DETECTION SYSTEM")
    print("   SUPERVISED CNN-LSTM CLASSIFIER")
    print("=" * 70)
    
    # Step 1: Load data
    train_df, test_df = load_data()
    
    # Step 2: Preprocess data
    (X_train, X_test, y_train, y_test, 
     attack_cat_train, attack_cat_test, 
     scaler, label_encoders, feature_columns) = preprocess_data(train_df, test_df)
    
    # Step 3: Create sequences
    SEQUENCE_LENGTH = 20
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = prepare_sequences(
        X_train, X_test, y_train, y_test, SEQUENCE_LENGTH
    )
    
    # Step 4: Split training data for validation
    X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(
        X_train_seq, y_train_seq, test_size=0.2, random_state=42, stratify=y_train_seq
    )
    
    print(f"📊 Data Shapes:")
    print(f"  Training: {X_train_seq.shape}")
    print(f"  Validation: {X_val_seq.shape}")
    print(f"  Testing: {X_test_seq.shape}")
    
    # Step 5: Build and train IMPROVED model
    n_features = X_train_seq.shape[2]
    model = build_improved_model(SEQUENCE_LENGTH, n_features)
    
    print(f"\n📐 Model Architecture Summary:")
    model.summary()
    
    # Step 6: Train the model
    history = train_improved_model(model, X_train_seq, y_train_seq, X_val_seq, y_val_seq)
    
    # Step 7: Evaluate
    metrics = evaluate_improved_model(model, X_test_seq, y_test_seq)
    
    # Step 8: Visualizations
    plot_confusion_matrix(y_test_seq, metrics['y_pred'], "Improved Model Confusion Matrix")
    plot_training_history(history)
    
    # Final summary
    print(f"\n" + "="*70)
    print("🎉 IMPROVED INTRUSION DETECTION SYSTEM SUMMARY")
    print("="*70)
    print(f"📊 Model Performance:")
    print(f"   • Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   • F1-Score:  {metrics['f1']:.4f}")
    print(f"   • AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"   • AUC-PR:    {metrics['auc_pr']:.4f}")
    print(f"\n🔧 System Configuration:")
    print(f"   • Sequence Length: {SEQUENCE_LENGTH}")
    print(f"   • Features: {n_features}")
    print(f"   • Architecture: CNN-LSTM Classifier (Supervised)")
    print(f"💡 The improved system is ready for deployment!")
    print("="*70)

# =============================================================================
# RUN THE SYSTEM
# =============================================================================

if __name__ == "__main__":
    main()