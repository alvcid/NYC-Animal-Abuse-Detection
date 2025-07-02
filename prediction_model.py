import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Para manejo de desbalance de clases
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from collections import Counter

# Para feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

def load_and_preprocess_data():
    """Cargar y preprocesar los datos"""
    
    print("=" * 70)
    print("MODELO DE PREDICCIÓN - NYC ANIMAL ABUSE")
    print("=" * 70)
    
    print("\n1. CARGANDO Y PREPARANDO DATOS...")
    print("-" * 40)
    
    # Cargar datos
    df = pd.read_csv('Animal_Abuse.csv/Animal_Abuse.csv')
    print(f"✓ Dataset cargado: {df.shape[0]:,} filas, {df.shape[1]} columnas")
    
    # Crear copia para trabajar
    data = df.copy()
    
    # Feature Engineering
    print("\n2. INGENIERÍA DE CARACTERÍSTICAS...")
    print("-" * 40)
    
    # Características temporales
    data['Created Date'] = pd.to_datetime(data['Created Date'], errors='coerce')
    data['Closed Date'] = pd.to_datetime(data['Closed Date'], errors='coerce')
    
    # Extraer características de fecha
    data['Hour'] = data['Created Date'].dt.hour
    data['DayOfWeek'] = data['Created Date'].dt.dayofweek
    data['Month'] = data['Created Date'].dt.month
    data['Quarter'] = data['Created Date'].dt.quarter
    data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Tiempo de resolución (si está disponible)
    data['Resolution_Time_Hours'] = (data['Closed Date'] - data['Created Date']).dt.total_seconds() / 3600
    
    # Características geográficas mejoradas
    data['Incident_Zip_Region'] = data['Incident Zip'].fillna(0).astype(int) // 100
    
    # Crear variable de densidad de casos por borough
    borough_counts = data['Borough'].value_counts()
    data['Borough_Case_Density'] = data['Borough'].map(borough_counts)
    
    # Características de ubicación
    data['Has_Specific_Address'] = (~data['Incident Address'].isna()).astype(int)
    data['Has_Coordinates'] = (~data['Latitude'].isna()).astype(int)
    
    print("✓ Características temporales creadas")
    print("✓ Características geográficas mejoradas")
    print("✓ Variables indicadoras agregadas")
    
    return data

def select_features_and_target(data):
    """Seleccionar características y variable objetivo"""
    
    print("\n3. SELECCIÓN DE CARACTERÍSTICAS...")
    print("-" * 40)
    
    # Variable objetivo
    target = 'Descriptor'
    
    # Seleccionar características más relevantes
    selected_features = [
        # Características temporales
        'Hour', 'DayOfWeek', 'Month', 'Quarter', 'IsWeekend',
        
        # Características geográficas
        'Borough', 'Incident_Zip_Region', 'Borough_Case_Density',
        
        # Características de ubicación
        'Location Type', 'Has_Specific_Address', 'Has_Coordinates',
        
        # Características del caso
        'Agency', 'Status',
        
        # Coordenadas (si están disponibles)
        'Latitude', 'Longitude',
        
        # Tiempo de resolución
        'Resolution_Time_Hours'
    ]
    
    # Filtrar características que existen en el dataset
    available_features = [col for col in selected_features if col in data.columns]
    
    print(f"✓ {len(available_features)} características seleccionadas:")
    for i, feature in enumerate(available_features, 1):
        missing_pct = (data[feature].isna().sum() / len(data)) * 100
        print(f"  {i:2d}. {feature} (faltantes: {missing_pct:.1f}%)")
    
    # Filtrar datos para características seleccionadas + target
    model_data = data[available_features + [target]].copy()
    
    # Eliminar filas con target faltante
    model_data = model_data.dropna(subset=[target])
    
    print(f"\n✓ Dataset final: {model_data.shape[0]:,} filas, {model_data.shape[1]-1} características")
    
    return model_data, available_features, target

def prepare_data_for_modeling(data, features, target):
    """Preparar datos para modelado"""
    
    print("\n4. PREPARACIÓN PARA MODELADO...")
    print("-" * 40)
    
    # Separar X e y
    X = data[features].copy()
    y = data[target].copy()
    
    # Análisis del desbalance de clases
    print("Distribución de clases original:")
    class_counts = y.value_counts()
    for class_name, count in class_counts.items():
        pct = (count / len(y)) * 100
        print(f"  - {class_name}: {count:,} ({pct:.2f}%)")
    
    # Identificar columnas categóricas y numéricas
    categorical_features = []
    numerical_features = []
    
    for col in X.columns:
        if X[col].dtype == 'object':
            categorical_features.append(col)
        else:
            numerical_features.append(col)
    
    print(f"\n✓ Características categóricas: {len(categorical_features)}")
    print(f"✓ Características numéricas: {len(numerical_features)}")
    
    # Crear preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✓ División train/test completada:")
    print(f"  - Train: {X_train.shape[0]:,} casos")
    print(f"  - Test: {X_test.shape[0]:,} casos")
    
    return X_train, X_test, y_train, y_test, preprocessor

def apply_balancing_techniques(X_train, y_train, method='smote'):
    """Aplicar técnicas de balanceamiento de clases"""
    
    print(f"\n5. BALANCEAMIENTO DE CLASES - {method.upper()}...")
    print("-" * 40)
    
    print("Distribución antes del balanceamiento:")
    original_distribution = Counter(y_train)
    for class_name, count in original_distribution.items():
        print(f"  - {class_name}: {count:,}")
    
    # Aplicar técnica de balanceamiento
    if method == 'smote':
        balancer = SMOTE(random_state=42, k_neighbors=3)
    elif method == 'random_oversample':
        balancer = RandomOverSampler(random_state=42)
    elif method == 'random_undersample':
        balancer = RandomUnderSampler(random_state=42)
    elif method == 'smoteenn':
        balancer = SMOTEENN(random_state=42)
    else:
        print("⚠️ Método no reconocido, usando SMOTE")
        balancer = SMOTE(random_state=42, k_neighbors=3)
    
    try:
        # Necesitamos convertir a formato numérico primero
        preprocessor_temp = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), [col for col in X_train.columns if X_train[col].dtype != 'object']),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), 
                 [col for col in X_train.columns if X_train[col].dtype == 'object'])
            ],
            remainder='passthrough'
        )
        
        X_train_processed = preprocessor_temp.fit_transform(X_train)
        X_train_balanced, y_train_balanced = balancer.fit_resample(X_train_processed, y_train)
        
        print("\nDistribución después del balanceamiento:")
        balanced_distribution = Counter(y_train_balanced)
        for class_name, count in balanced_distribution.items():
            print(f"  - {class_name}: {count:,}")
        
        print(f"✓ Balanceamiento completado: {len(y_train_balanced):,} casos")
        
        return X_train_balanced, y_train_balanced, preprocessor_temp
        
    except Exception as e:
        print(f"✗ Error en balanceamiento: {e}")
        print("Continuando sin balanceamiento...")
        return None, None, None

def train_multiple_models(X_train, X_test, y_train, y_test, preprocessor):
    """Entrenar múltiples modelos y compararlos"""
    
    print("\n6. ENTRENAMIENTO DE MODELOS...")
    print("-" * 40)
    
    # Definir modelos a probar
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Naive Bayes': GaussianNB()
    }
    
    # Almacenar resultados
    results = {}
    
    # Entrenar cada modelo
    for name, model in models.items():
        print(f"\n🔄 Entrenando {name}...")
        
        try:
            # Crear pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            # Entrenar
            pipeline.fit(X_train, y_train)
            
            # Predecir
            y_pred = pipeline.predict(X_test)
            
            # Calcular métricas
            accuracy = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            
            # Guardar resultados
            results[name] = {
                'model': pipeline,
                'accuracy': accuracy,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro,
                'predictions': y_pred
            }
            
            print(f"✓ Accuracy: {accuracy:.4f}")
            print(f"✓ F1-Score (weighted): {f1_weighted:.4f}")
            print(f"✓ F1-Score (macro): {f1_macro:.4f}")
            
        except Exception as e:
            print(f"✗ Error entrenando {name}: {e}")
            continue
    
    return results

def evaluate_best_model(results, y_test):
    """Evaluar el mejor modelo"""
    
    print("\n7. EVALUACIÓN DEL MEJOR MODELO...")
    print("-" * 40)
    
    # Encontrar el mejor modelo por F1-score weighted
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_weighted'])
    best_model_info = results[best_model_name]
    
    print(f"🏆 MEJOR MODELO: {best_model_name}")
    print(f"   Accuracy: {best_model_info['accuracy']:.4f}")
    print(f"   F1-Score (weighted): {best_model_info['f1_weighted']:.4f}")
    print(f"   F1-Score (macro): {best_model_info['f1_macro']:.4f}")
    
    # Reporte de clasificación detallado
    print(f"\n📊 REPORTE DETALLADO - {best_model_name}:")
    print("-" * 50)
    y_pred_best = best_model_info['predictions']
    print(classification_report(y_test, y_pred_best))
    
    return best_model_name, best_model_info

def create_visualizations(results, y_test):
    """Crear visualizaciones de resultados"""
    
    print("\n8. CREANDO VISUALIZACIONES...")
    print("-" * 40)
    
    # Configurar figura
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Comparación de modelos
    plt.subplot(2, 3, 1)
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    f1_scores = [results[name]['f1_weighted'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    plt.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
    plt.xlabel('Modelos')
    plt.ylabel('Score')
    plt.title('Comparación de Rendimiento de Modelos')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Matriz de confusión del mejor modelo
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_weighted'])
    y_pred_best = results[best_model_name]['predictions']
    
    plt.subplot(2, 3, 2)
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {best_model_name}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    
    # 3. Distribución de predicciones
    plt.subplot(2, 3, 3)
    pred_counts = pd.Series(y_pred_best).value_counts()
    plt.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%')
    plt.title('Distribución de Predicciones')
    
    # 4. F1-Score por clase
    plt.subplot(2, 3, 4)
    from sklearn.metrics import f1_score
    classes = sorted(y_test.unique())
    f1_per_class = []
    
    for class_name in classes:
        f1 = f1_score(y_test == class_name, y_pred_best == class_name)
        f1_per_class.append(f1)
    
    plt.barh(classes, f1_per_class)
    plt.xlabel('F1-Score')
    plt.title('F1-Score por Clase')
    plt.grid(True, alpha=0.3)
    
    # 5. Comparación de métricas
    plt.subplot(2, 3, 5)
    metrics_data = []
    for name in model_names:
        metrics_data.append([
            results[name]['accuracy'],
            results[name]['f1_weighted'],
            results[name]['f1_macro']
        ])
    
    metrics_df = pd.DataFrame(metrics_data, 
                             columns=['Accuracy', 'F1-Weighted', 'F1-Macro'],
                             index=model_names)
    
    metrics_df.plot(kind='bar', ax=plt.gca())
    plt.title('Comparación de Todas las Métricas')
    plt.xlabel('Modelos')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Distribución de errores
    plt.subplot(2, 3, 6)
    errors = y_test != y_pred_best
    error_by_class = pd.Series(y_test[errors]).value_counts()
    plt.bar(error_by_class.index, error_by_class.values, color='red', alpha=0.7)
    plt.title('Errores por Clase Real')
    plt.xlabel('Clase Real')
    plt.ylabel('Número de Errores')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Visualizaciones guardadas como 'model_evaluation.png'")

def main():
    """Función principal del pipeline de ML"""
    
    try:
        # 1. Cargar y preprocesar datos
        data = load_and_preprocess_data()
        
        # 2. Seleccionar características
        model_data, features, target = select_features_and_target(data)
        
        # 3. Preparar datos para modelado
        X_train, X_test, y_train, y_test, preprocessor = prepare_data_for_modeling(
            model_data, features, target
        )
        
        # 4. Intentar balanceamiento de clases
        X_train_balanced, y_train_balanced, preprocessor_balanced = apply_balancing_techniques(
            X_train, y_train, method='smote'
        )
        
        # 5. Entrenar modelos (usar datos balanceados si están disponibles)
        if X_train_balanced is not None:
            print("\n🔄 Usando datos balanceados para entrenamiento...")
            # Para datos balanceados, necesitamos ajustar el enfoque
            # Entrenaremos con el preprocessor ya aplicado
            models_balanced = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'Naive Bayes': GaussianNB()
            }
            
            results = {}
            X_test_processed = preprocessor_balanced.transform(X_test)
            
            for name, model in models_balanced.items():
                print(f"\n🔄 Entrenando {name} con datos balanceados...")
                
                try:
                    model.fit(X_train_balanced, y_train_balanced)
                    y_pred = model.predict(X_test_processed)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    f1_weighted = f1_score(y_test, y_pred, average='weighted')
                    f1_macro = f1_score(y_test, y_pred, average='macro')
                    
                    results[name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'f1_weighted': f1_weighted,
                        'f1_macro': f1_macro,
                        'predictions': y_pred
                    }
                    
                    print(f"✓ Accuracy: {accuracy:.4f}")
                    print(f"✓ F1-Score (weighted): {f1_weighted:.4f}")
                    
                except Exception as e:
                    print(f"✗ Error entrenando {name}: {e}")
                    continue
        else:
            # Entrenar con datos originales
            results = train_multiple_models(X_train, X_test, y_train, y_test, preprocessor)
        
        # 6. Evaluar mejor modelo
        if results:
            best_model_name, best_model_info = evaluate_best_model(results, y_test)
            
            # 7. Crear visualizaciones
            create_visualizations(results, y_test)
            
            # 8. Resumen final
            print("\n" + "=" * 70)
            print("RESUMEN FINAL DEL MODELO DE PREDICCIÓN")
            print("=" * 70)
            print(f"🏆 Mejor modelo: {best_model_name}")
            print(f"📊 Accuracy final: {best_model_info['accuracy']:.4f}")
            print(f"📊 F1-Score (weighted): {best_model_info['f1_weighted']:.4f}")
            print(f"📊 F1-Score (macro): {best_model_info['f1_macro']:.4f}")
            print("\n✅ Pipeline completado exitosamente!")
            print("✅ Modelo listo para predicciones en producción")
            
            return best_model_info['model'], results
        else:
            print("✗ No se pudieron entrenar modelos exitosamente")
            return None, None
            
    except Exception as e:
        print(f"✗ Error en el pipeline: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None

if __name__ == "__main__":
    best_model, all_results = main() 