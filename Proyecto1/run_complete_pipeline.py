#!/usr/bin/env python3
"""
Pipeline Completo de Machine Learning - Proyecto 1
Ejecuta todo el flujo: EDA → Preprocessing → Benchmarking → Metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler, 
                                 LabelEncoder, OneHotEncoder, OrdinalEncoder)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve)
import time
import os

warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Cargar y preparar el dataset"""
    print("📊 CARGANDO DATASET...")
    
    df = pd.read_csv('retail_sales_dataset.csv')
    
    # Convertir Date a datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extraer características temporales
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.strftime('%B')
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayName'] = df['Date'].dt.strftime('%A')
    df['Quarter'] = df['Date'].dt.quarter
    
    # Clasificación de ventas
    def clasificador_ventas(amount):
        if amount >= 1000:
            return 'Alta'
        elif amount >= 300:
            return 'Media'
        else:
            return 'Baja'
    
    df['Sales_Category'] = df['Total Amount'].apply(clasificador_ventas)
    
    # Grupos de edad
    def grupo_edad(age):
        if age < 25:
            return 'Joven'
        elif age < 50:
            return 'Adulto'
        else:
            return 'Senior'
    
    df['Age_Group'] = df['Age'].apply(grupo_edad)
    
    print(f"✅ Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
    return df

def create_preprocessors():
    """Crear pipelines de preprocesamiento"""
    print("🔧 CREANDO PIPELINES DE PREPROCESAMIENTO...")
    
    # Definir features
    exclude_columns = [
        'Transaction ID', 'Customer ID', 'Date',
        'Total Amount', 'Sales_Category'
    ]
    
    # Datos de ejemplo para obtener tipos de columnas
    df_sample = load_and_prepare_data()
    features_all = [col for col in df_sample.columns if col not in exclude_columns]
    numeric_features = df_sample[features_all].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df_sample[features_all].select_dtypes(include=['object']).columns.tolist()
    
    # Pipeline 1: StandardScaler + OneHotEncoder
    numeric_pipeline_1 = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline_1 = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor_1 = ColumnTransformer([
        ('numeric', numeric_pipeline_1, numeric_features),
        ('categorical', categorical_pipeline_1, categorical_features)
    ])
    
    # Pipeline 2: MinMaxScaler + OneHotEncoder
    numeric_pipeline_2 = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    
    preprocessor_2 = ColumnTransformer([
        ('numeric', numeric_pipeline_2, numeric_features),
        ('categorical', categorical_pipeline_1, categorical_features)
    ])
    
    # Pipeline 3: RobustScaler + OrdinalEncoder
    categorical_pipeline_3 = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    numeric_pipeline_3 = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    preprocessor_3 = ColumnTransformer([
        ('numeric', numeric_pipeline_3, numeric_features),
        ('categorical', categorical_pipeline_3, categorical_features)
    ])
    
    preprocessors = {
        'StandardScaler_OneHot': preprocessor_1,
        'MinMaxScaler_OneHot': preprocessor_2,
        'RobustScaler_Ordinal': preprocessor_3
    }
    
    print(f"✅ {len(preprocessors)} pipelines creados")
    return preprocessors, features_all, numeric_features, categorical_features

def preprocess_data():
    """Preprocesar datos con todos los pipelines"""
    print("🔄 PREPROCESANDO DATOS...")
    
    df = load_and_prepare_data()
    preprocessors, features_all, numeric_features, categorical_features = create_preprocessors()
    
    # Preparar X y y
    X = df[features_all]
    y = df['Sales_Category']
    
    # Codificar target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Aplicar cada pipeline
    processed_datasets = {}
    
    for name, preprocessor in preprocessors.items():
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        processed_datasets[name] = {
            'X_train': X_train_processed,
            'X_test': X_test_processed,
            'y_train': y_train,
            'y_test': y_test,
            'preprocessor': preprocessor,
            'label_encoder': label_encoder
        }
    
    # Guardar todo
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    for name, preprocessor in preprocessors.items():
        joblib.dump(preprocessor, f'models/preprocessor_{name}.joblib')
    
    joblib.dump(label_encoder, 'models/label_encoder.joblib')
    joblib.dump(processed_datasets, 'models/processed_datasets.joblib')
    
    print(f"✅ Datos preprocesados y guardados")
    return processed_datasets, label_encoder

def benchmark_models():
    """Evaluar múltiples modelos ML"""
    print("🤖 BENCHMARKING DE MODELOS...")
    
    # Cargar datos preprocesados
    processed_datasets = joblib.load('models/processed_datasets.joblib')
    label_encoder = joblib.load('models/label_encoder.joblib')
    
    # Usar dataset principal
    main_dataset = processed_datasets['StandardScaler_OneHot']
    X_train = main_dataset['X_train']
    X_test = main_dataset['X_test']
    y_train = main_dataset['y_train']
    y_test = main_dataset['y_test']
    
    # Definir modelos
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Support Vector Machine': SVC(random_state=42, probability=True),
        'Naive Bayes': GaussianNB()
    }
    
    # Evaluar con validación cruzada
    cv_results = {}
    training_times = {}
    cv_folds = 5
    cv_scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for model_name, model in models.items():
        print(f"   🚀 Evaluando: {model_name}")
        
        start_time = time.time()
        
        cv_results[model_name] = {}
        for metric in cv_scoring:
            scores = cross_val_score(model, X_train, y_train, 
                                   cv=skf, scoring=metric, n_jobs=-1)
            cv_results[model_name][metric] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std()
            }
        
        training_times[model_name] = time.time() - start_time
    
    # Crear tabla de resultados
    results_data = []
    for model_name in cv_results.keys():
        row = {
            'Model': model_name,
            'Accuracy_Mean': cv_results[model_name]['accuracy']['mean'],
            'F1_Mean': cv_results[model_name]['f1_macro']['mean'],
            'Training_Time': training_times[model_name]
        }
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values('F1_Mean', ascending=False).reset_index(drop=True)
    
    # Mejor modelo
    best_model_name = results_df.iloc[0]['Model']
    best_model = models[best_model_name]
    
    # Entrenar mejor modelo
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    
    # Métricas finales
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro')
    test_precision = precision_score(y_test, y_pred, average='macro')
    test_recall = recall_score(y_test, y_pred, average='macro')
    
    # Guardar resultados
    joblib.dump(best_model, f'models/best_model_{best_model_name.replace(" ", "_")}.joblib')
    
    test_results = {
        'model_name': best_model_name,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'cv_accuracy': cv_results[best_model_name]['accuracy']['mean'],
        'cv_f1': cv_results[best_model_name]['f1_macro']['mean']
    }
    
    joblib.dump(test_results, 'models/test_results.joblib')
    results_df.to_csv('reports/cv_results_comparison.csv', index=False)
    
    print(f"🏆 Mejor modelo: {best_model_name}")
    print(f"   📊 Accuracy: {test_accuracy:.4f}")
    print(f"   📊 F1-Score: {test_f1:.4f}")
    
    return best_model_name, test_results

def generate_final_metrics():
    """Generar métricas finales y reportes"""
    print("📈 GENERANDO ANÁLISIS DE MÉTRICAS...")
    
    # Cargar modelo y datos
    test_results = joblib.load('models/test_results.joblib')
    processed_datasets = joblib.load('models/processed_datasets.joblib')
    label_encoder = joblib.load('models/label_encoder.joblib')
    
    best_model_name = test_results['model_name']
    model_filename = f'models/best_model_{best_model_name.replace(" ", "_")}.joblib'
    best_model = joblib.load(model_filename)
    
    # Datos de prueba
    main_dataset = processed_datasets['StandardScaler_OneHot']
    X_test = main_dataset['X_test']
    y_test = main_dataset['y_test']
    
    # Predicciones
    y_pred = best_model.predict(X_test)
    
    # Reporte de clasificación
    class_names = label_encoder.classes_
    classification_rep = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True
    )
    
    # Guardar reporte
    report_df = pd.DataFrame(classification_rep).transpose()
    report_df.to_csv('reports/classification_report.csv')
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    np.savetxt('reports/confusion_matrix.csv', cm, delimiter=',', fmt='%d')
    
    # Guardar reporte en texto
    with open('reports/classification_report.txt', 'w') as f:
        f.write(f"Reporte de Clasificación - Modelo: {best_model_name}\n")
        f.write("="*60 + "\n\n")
        f.write(classification_report(y_test, y_pred, target_names=class_names))
    
    print("✅ Reportes de métricas generados")
    return True

def main():
    """Ejecutar pipeline completo"""
    print("🚀 EJECUTANDO PIPELINE COMPLETO DE MACHINE LEARNING")
    print("="*60)
    
    try:
        # 1. Preprocesamiento
        processed_datasets, label_encoder = preprocess_data()
        
        # 2. Benchmarking
        best_model_name, test_results = benchmark_models()
        
        # 3. Métricas
        generate_final_metrics()
        
        print("\n✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*40)
        print(f"🏆 Mejor modelo: {best_model_name}")
        print(f"📊 Accuracy: {test_results['test_accuracy']:.4f}")
        print(f"📊 F1-Score: {test_results['test_f1']:.4f}")
        print("\n📁 Archivos generados:")
        print("   • models/ - Modelos entrenados y preprocessors")
        print("   • reports/ - Reportes y métricas detalladas")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en el pipeline: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 ¡Proyecto listo para GitHub!")
    else:
        print("\n❌ Revisar errores en el pipeline") 