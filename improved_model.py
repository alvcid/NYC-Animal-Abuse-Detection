import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier  # Maneja NaN nativamente
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Para manejo de desbalance de clases
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter

# Para optimización
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

def enhanced_data_preprocessing():
    """Preprocesamiento mejorado de datos con manejo robusto de valores faltantes"""
    
    print("=" * 75)
    print("MODELO MEJORADO DE PREDICCIÓN - NYC ANIMAL ABUSE")
    print("=" * 75)
    
    print("\n1. CARGA Y PREPROCESAMIENTO MEJORADO DE DATOS...")
    print("-" * 50)
    
    # Cargar datos
    df = pd.read_csv('Animal_Abuse.csv/Animal_Abuse.csv')
    print(f"✓ Dataset cargado: {df.shape[0]:,} filas, {df.shape[1]} columnas")
    
    # Crear copia para trabajar
    data = df.copy()
    
    # Análisis detallado de valores faltantes
    print("\n2. ANÁLISIS DETALLADO DE VALORES FALTANTES...")
    print("-" * 50)
    
    missing_analysis = []
    for col in data.columns:
        missing_count = data[col].isnull().sum()
        missing_pct = (missing_count / len(data)) * 100
        dtype = str(data[col].dtype)
        unique_values = data[col].nunique()
        
        missing_analysis.append({
            'Column': col,
            'Missing_Count': missing_count,
            'Missing_Pct': missing_pct,
            'Data_Type': dtype,
            'Unique_Values': unique_values
        })
    
    missing_df = pd.DataFrame(missing_analysis)
    missing_df = missing_df.sort_values('Missing_Pct', ascending=False)
    
    print("Top 10 columnas con más valores faltantes:")
    print(missing_df.head(10)[['Column', 'Missing_Pct']].to_string(index=False))
    
    # Feature Engineering Mejorado
    print("\n3. INGENIERÍA DE CARACTERÍSTICAS MEJORADA...")
    print("-" * 50)
    
    # Características temporales
    data['Created Date'] = pd.to_datetime(data['Created Date'], errors='coerce')
    data['Closed Date'] = pd.to_datetime(data['Closed Date'], errors='coerce')
    
    # Extraer características de fecha más detalladas
    data['Hour'] = data['Created Date'].dt.hour
    data['DayOfWeek'] = data['Created Date'].dt.dayofweek
    data['Month'] = data['Created Date'].dt.month
    data['Quarter'] = data['Created Date'].dt.quarter
    data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)
    data['IsBusinessHour'] = data['Hour'].between(9, 17).astype(int)
    data['IsNightTime'] = data['Hour'].between(22, 6).astype(int)
    
    # Tiempo de resolución mejorado
    data['Resolution_Time_Hours'] = (data['Closed Date'] - data['Created Date']).dt.total_seconds() / 3600
    data['Resolution_Time_Days'] = data['Resolution_Time_Hours'] / 24
    data['Has_Resolution_Time'] = (~data['Resolution_Time_Hours'].isna()).astype(int)
    
    # Características geográficas mejoradas
    data['Incident_Zip_Valid'] = data['Incident Zip'].notna().astype(int)
    data['Incident_Zip_Region'] = data['Incident Zip'].fillna(0).astype(int) // 100
    
    # Densidad de casos por área
    borough_counts = data['Borough'].value_counts()
    data['Borough_Case_Density'] = data['Borough'].map(borough_counts)
    
    # Características de ubicación mejoradas
    data['Has_Specific_Address'] = (~data['Incident Address'].isna()).astype(int)
    data['Has_Coordinates'] = (~data['Latitude'].isna()).astype(int)
    data['Has_Street_Info'] = (~data['Street Name'].isna()).astype(int)
    data['Has_Cross_Streets'] = ((~data['Cross Street 1'].isna()) & (~data['Cross Street 2'].isna())).astype(int)
    
    # Características de tipo de ubicación
    location_type_counts = data['Location Type'].value_counts()
    data['Location_Type_Frequency'] = data['Location Type'].map(location_type_counts).fillna(0)
    
    # Características de agencia
    agency_counts = data['Agency'].value_counts()
    data['Agency_Case_Volume'] = data['Agency'].map(agency_counts)
    
    print("✓ Características temporales avanzadas creadas")
    print("✓ Características geográficas mejoradas")
    print("✓ Variables indicadoras de calidad de datos")
    print("✓ Características de frecuencia/densidad")
    
    return data

def robust_feature_selection(data):
    """Selección robusta de características"""
    
    print("\n4. SELECCIÓN ROBUSTA DE CARACTERÍSTICAS...")
    print("-" * 50)
    
    # Variable objetivo
    target = 'Descriptor'
    
    # Seleccionar características más robustas
    selected_features = [
        # Características temporales
        'Hour', 'DayOfWeek', 'Month', 'Quarter', 'IsWeekend', 'IsBusinessHour', 'IsNightTime',
        
        # Características geográficas
        'Borough', 'Incident_Zip_Region', 'Borough_Case_Density',
        
        # Características de ubicación
        'Has_Specific_Address', 'Has_Coordinates', 'Has_Street_Info', 'Has_Cross_Streets',
        'Location_Type_Frequency',
        
        # Características del caso
        'Agency_Case_Volume', 'Has_Resolution_Time',
        
        # Coordenadas (manejadas con cuidado)
        'Latitude', 'Longitude',
        
        # Tiempo de resolución
        'Resolution_Time_Hours', 'Resolution_Time_Days'
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

def advanced_preprocessing_pipeline(X, y):
    """Pipeline avanzado de preprocesamiento"""
    
    print("\n5. PIPELINE AVANZADO DE PREPROCESAMIENTO...")
    print("-" * 50)
    
    # Identificar tipos de columnas
    categorical_features = []
    numerical_features = []
    
    for col in X.columns:
        if X[col].dtype == 'object':
            categorical_features.append(col)
        else:
            numerical_features.append(col)
    
    print(f"✓ Características categóricas: {len(categorical_features)}")
    print(f"✓ Características numéricas: {len(numerical_features)}")
    
    # Crear pipelines específicos para cada tipo de dato
    numerical_pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),  # Imputer más sofisticado
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    
    # Combinar pipelines
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    # Dividir en train/test con estratificación
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✓ División train/test completada:")
    print(f"  - Train: {X_train.shape[0]:,} casos")
    print(f"  - Test: {X_test.shape[0]:,} casos")
    
    # Análisis de distribución de clases
    print("\nDistribución de clases en train:")
    class_counts = y_train.value_counts()
    for class_name, count in class_counts.items():
        pct = (count / len(y_train)) * 100
        print(f"  - {class_name}: {count:,} ({pct:.2f}%)")
    
    return X_train, X_test, y_train, y_test, preprocessor

def robust_class_balancing(X_train, y_train, preprocessor):
    """Balanceamiento robusto de clases"""
    
    print("\n6. BALANCEAMIENTO ROBUSTO DE CLASES...")
    print("-" * 50)
    
    try:
        # Aplicar preprocesamiento primero
        X_train_processed = preprocessor.fit_transform(X_train)
        
        print("Distribución antes del balanceamiento:")
        original_distribution = Counter(y_train)
        for class_name, count in original_distribution.items():
            print(f"  - {class_name}: {count:,}")
        
        # Usar ADASYN que es más robusto que SMOTE
        balancer = ADASYN(random_state=42, n_neighbors=3)
        
        X_train_balanced, y_train_balanced = balancer.fit_resample(X_train_processed, y_train)
        
        print("\nDistribución después del balanceamiento:")
        balanced_distribution = Counter(y_train_balanced)
        for class_name, count in balanced_distribution.items():
            print(f"  - {class_name}: {count:,}")
        
        print(f"✓ Balanceamiento completado: {len(y_train_balanced):,} casos")
        
        return X_train_balanced, y_train_balanced, True
        
    except Exception as e:
        print(f"⚠️ Error en balanceamiento: {e}")
        print("Usando estrategia de pesos balanceados en su lugar...")
        return None, None, False

def train_optimized_models(X_train, X_test, y_train, y_test, preprocessor, use_balanced_data=False, X_train_balanced=None, y_train_balanced=None):
    """Entrenar modelos optimizados con búsqueda de hiperparámetros"""
    
    print("\n7. ENTRENAMIENTO DE MODELOS OPTIMIZADOS...")
    print("-" * 50)
    
    # Definir modelos con parámetros optimizados
    models = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'Hist Gradient Boosting': {  # Maneja NaN nativamente
            'model': HistGradientBoostingClassifier(random_state=42, class_weight='balanced'),
            'params': {
                'max_iter': [100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        },
        'Extra Trees': {
            'model': ExtraTreesClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10]
            }
        }
    }
    
    results = {}
    
    for name, model_config in models.items():
        print(f"\n🔄 Optimizando {name}...")
        
        try:
            if use_balanced_data and X_train_balanced is not None:
                # Usar datos balanceados
                X_train_use = X_train_balanced
                y_train_use = y_train_balanced
                X_test_processed = preprocessor.transform(X_test)
                
                # Búsqueda de hiperparámetros
                search = RandomizedSearchCV(
                    model_config['model'],
                    model_config['params'],
                    n_iter=20,
                    cv=3,
                    scoring='f1_weighted',
                    random_state=42,
                    n_jobs=-1
                )
                
                search.fit(X_train_use, y_train_use)
                best_model = search.best_estimator_
                
                # Predecir en test
                y_pred = best_model.predict(X_test_processed)
                
            else:
                # Usar pipeline normal
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', model_config['model'])
                ])
                
                # Parámetros para pipeline
                pipeline_params = {}
                for key, value in model_config['params'].items():
                    pipeline_params[f'classifier__{key}'] = value
                
                # Búsqueda de hiperparámetros
                search = RandomizedSearchCV(
                    pipeline,
                    pipeline_params,
                    n_iter=20,
                    cv=3,
                    scoring='f1_weighted',
                    random_state=42,
                    n_jobs=-1
                )
                
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                
                # Predecir en test
                y_pred = best_model.predict(X_test)
            
            # Calcular métricas
            accuracy = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            
            # Guardar resultados
            results[name] = {
                'model': best_model,
                'best_params': search.best_params_,
                'accuracy': accuracy,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro,
                'predictions': y_pred,
                'cv_score': search.best_score_
            }
            
            print(f"✓ Mejores parámetros: {search.best_params_}")
            print(f"✓ CV Score: {search.best_score_:.4f}")
            print(f"✓ Test Accuracy: {accuracy:.4f}")
            print(f"✓ Test F1-Score (weighted): {f1_weighted:.4f}")
            
        except Exception as e:
            print(f"✗ Error optimizando {name}: {e}")
            continue
    
    return results

def comprehensive_evaluation(results, y_test):
    """Evaluación comprensiva del modelo"""
    
    print("\n8. EVALUACIÓN COMPRENSIVA...")
    print("-" * 50)
    
    if not results:
        print("✗ No hay resultados para evaluar")
        return None, None
    
    # Encontrar el mejor modelo
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_weighted'])
    best_model_info = results[best_model_name]
    
    print(f"🏆 MEJOR MODELO: {best_model_name}")
    print(f"   CV Score: {best_model_info['cv_score']:.4f}")
    print(f"   Test Accuracy: {best_model_info['accuracy']:.4f}")
    print(f"   Test F1-Score (weighted): {best_model_info['f1_weighted']:.4f}")
    print(f"   Test F1-Score (macro): {best_model_info['f1_macro']:.4f}")
    print(f"   Mejores parámetros: {best_model_info['best_params']}")
    
    # Reporte de clasificación detallado
    print(f"\n📊 REPORTE DETALLADO - {best_model_name}:")
    print("-" * 60)
    y_pred_best = best_model_info['predictions']
    print(classification_report(y_test, y_pred_best))
    
    # Análisis por clase
    print("\n📈 ANÁLISIS POR CLASE:")
    print("-" * 30)
    classes = sorted(y_test.unique())
    for class_name in classes:
        class_mask = y_test == class_name
        class_pred_mask = y_pred_best == class_name
        
        true_positives = np.sum(class_mask & class_pred_mask)
        false_positives = np.sum(~class_mask & class_pred_mask)
        false_negatives = np.sum(class_mask & ~class_pred_mask)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        print(f"  {class_name}:")
        print(f"    - Casos reales: {np.sum(class_mask)}")
        print(f"    - Casos predichos: {np.sum(class_pred_mask)}")
        print(f"    - Precisión: {precision:.3f}")
        print(f"    - Recall: {recall:.3f}")
    
    return best_model_name, best_model_info

def create_advanced_visualizations(results, y_test):
    """Crear visualizaciones avanzadas"""
    
    print("\n9. CREANDO VISUALIZACIONES AVANZADAS...")
    print("-" * 50)
    
    # Configurar figura más grande
    fig = plt.figure(figsize=(25, 20))
    
    # 1. Comparación de modelos (métricas múltiples)
    plt.subplot(3, 4, 1)
    model_names = list(results.keys())
    metrics = ['accuracy', 'f1_weighted', 'f1_macro', 'cv_score']
    
    x = np.arange(len(model_names))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[name][metric] for name in model_names]
        plt.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
    
    plt.xlabel('Modelos')
    plt.ylabel('Score')
    plt.title('Comparación Completa de Modelos')
    plt.xticks(x + width*1.5, model_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Matriz de confusión mejorada
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_weighted'])
    y_pred_best = results[best_model_name]['predictions']
    
    plt.subplot(3, 4, 2)
    cm = confusion_matrix(y_test, y_pred_best)
    classes = sorted(y_test.unique())
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Matriz de Confusión - {best_model_name}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # 3. Distribución de predicciones vs real
    plt.subplot(3, 4, 3)
    real_counts = pd.Series(y_test).value_counts()
    pred_counts = pd.Series(y_pred_best).value_counts()
    
    comparison_df = pd.DataFrame({
        'Real': real_counts,
        'Predicho': pred_counts
    }).fillna(0)
    
    comparison_df.plot(kind='bar', ax=plt.gca())
    plt.title('Distribución: Real vs Predicho')
    plt.xlabel('Clases')
    plt.ylabel('Número de Casos')
    plt.xticks(rotation=45)
    plt.legend()
    
    # 4. F1-Score por clase
    plt.subplot(3, 4, 4)
    f1_per_class = []
    for class_name in classes:
        f1 = f1_score(y_test == class_name, y_pred_best == class_name)
        f1_per_class.append(f1)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
    bars = plt.barh(classes, f1_per_class, color=colors)
    plt.xlabel('F1-Score')
    plt.title('F1-Score por Clase')
    plt.grid(True, alpha=0.3)
    
    # Añadir valores en las barras
    for i, (bar, score) in enumerate(zip(bars, f1_per_class)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center', fontsize=8)
    
    # 5. Precision-Recall por clase
    plt.subplot(3, 4, 5)
    precision_per_class = []
    recall_per_class = []
    
    for class_name in classes:
        class_mask = y_test == class_name
        pred_mask = y_pred_best == class_name
        
        # Calcular precision y recall
        tp = np.sum(class_mask & pred_mask)
        fp = np.sum(~class_mask & pred_mask)
        fn = np.sum(class_mask & ~pred_mask)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precision_per_class.append(precision)
        recall_per_class.append(recall)
    
    x = np.arange(len(classes))
    width = 0.35
    
    plt.bar(x - width/2, precision_per_class, width, label='Precision', alpha=0.8)
    plt.bar(x + width/2, recall_per_class, width, label='Recall', alpha=0.8)
    plt.xlabel('Clases')
    plt.ylabel('Score')
    plt.title('Precision vs Recall por Clase')
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Evolución de CV scores
    plt.subplot(3, 4, 6)
    cv_scores = [results[name]['cv_score'] for name in model_names]
    test_scores = [results[name]['f1_weighted'] for name in model_names]
    
    x = np.arange(len(model_names))
    plt.scatter(cv_scores, test_scores, s=100, alpha=0.7)
    
    for i, name in enumerate(model_names):
        plt.annotate(name, (cv_scores[i], test_scores[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('CV Score (F1-weighted)')
    plt.ylabel('Test Score (F1-weighted)')
    plt.title('CV vs Test Performance')
    plt.grid(True, alpha=0.3)
    
    # Línea diagonal para referencia
    min_val = min(min(cv_scores), min(test_scores))
    max_val = max(max(cv_scores), max(test_scores))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('improved_model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Visualizaciones avanzadas guardadas como 'improved_model_evaluation.png'")

def main():
    """Función principal del pipeline mejorado"""
    
    try:
        # 1. Preprocesamiento mejorado
        data = enhanced_data_preprocessing()
        
        # 2. Selección robusta de características
        model_data, features, target = robust_feature_selection(data)
        
        # 3. Pipeline avanzado de preprocesamiento
        X = model_data[features]
        y = model_data[target]
        
        X_train, X_test, y_train, y_test, preprocessor = advanced_preprocessing_pipeline(X, y)
        
        # 4. Balanceamiento robusto de clases
        X_train_balanced, y_train_balanced, balance_success = robust_class_balancing(
            X_train, y_train, preprocessor
        )
        
        # 5. Entrenamiento de modelos optimizados
        results = train_optimized_models(
            X_train, X_test, y_train, y_test, preprocessor, 
            balance_success, X_train_balanced, y_train_balanced
        )
        
        # 6. Evaluación comprensiva
        if results:
            best_model_name, best_model_info = comprehensive_evaluation(results, y_test)
            
            # 7. Visualizaciones avanzadas
            create_advanced_visualizations(results, y_test)
            
            # 8. Resumen final mejorado
            print("\n" + "=" * 75)
            print("RESUMEN FINAL DEL MODELO MEJORADO")
            print("=" * 75)
            print(f"🏆 Mejor modelo: {best_model_name}")
            print(f"📊 CV Score: {best_model_info['cv_score']:.4f}")
            print(f"📊 Test Accuracy: {best_model_info['accuracy']:.4f}")
            print(f"📊 Test F1-Score (weighted): {best_model_info['f1_weighted']:.4f}")
            print(f"📊 Test F1-Score (macro): {best_model_info['f1_macro']:.4f}")
            print(f"⚙️  Mejores parámetros: {best_model_info['best_params']}")
            
            # Comparar con baseline
            print(f"\n📈 MEJORAS RESPECTO AL MODELO ANTERIOR:")
            print(f"   - Accuracy: {best_model_info['accuracy']:.4f} vs 0.5652 (anterior)")
            print(f"   - F1-weighted: {best_model_info['f1_weighted']:.4f} vs 0.5057 (anterior)")
            
            improvement_acc = (best_model_info['accuracy'] - 0.5652) / 0.5652 * 100
            improvement_f1 = (best_model_info['f1_weighted'] - 0.5057) / 0.5057 * 100
            
            print(f"   - Mejora en Accuracy: {improvement_acc:+.1f}%")
            print(f"   - Mejora en F1-weighted: {improvement_f1:+.1f}%")
            
            print("\n✅ Pipeline mejorado completado exitosamente!")
            print("✅ Modelo optimizado listo para producción")
            
            return best_model_info['model'], results
        else:
            print("✗ No se pudieron entrenar modelos exitosamente")
            return None, None
            
    except Exception as e:
        print(f"✗ Error en el pipeline mejorado: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None

if __name__ == "__main__":
    best_model, all_results = main() 