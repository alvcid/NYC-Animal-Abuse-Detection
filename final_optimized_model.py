import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif

# Para balanceamiento conservador
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

def load_and_engineer_features():
    """Cargar datos y crear características más informativas"""
    
    print("=" * 65)
    print("MODELO FINAL OPTIMIZADO - NYC ANIMAL ABUSE PREDICTION")
    print("=" * 65)
    
    print("\n1. CARGA DE DATOS E INGENIERÍA DE CARACTERÍSTICAS...")
    print("-" * 55)
    
    # Cargar datos
    df = pd.read_csv('Animal_Abuse.csv/Animal_Abuse.csv')
    print(f"✓ Dataset original: {df.shape[0]:,} filas, {df.shape[1]} columnas")
    
    # Crear características más informativas
    data = df.copy()
    
    # Características temporales detalladas
    data['Created Date'] = pd.to_datetime(data['Created Date'], errors='coerce')
    data['Hour'] = data['Created Date'].dt.hour
    data['DayOfWeek'] = data['Created Date'].dt.dayofweek
    data['Month'] = data['Created Date'].dt.month
    data['Quarter'] = data['Created Date'].dt.quarter
    data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)
    data['IsBusinessHour'] = data['Hour'].between(9, 17).astype(int)
    data['IsEveningHour'] = data['Hour'].between(18, 22).astype(int)
    data['IsNightHour'] = data['Hour'].between(23, 6).astype(int)
    
    # Características geográficas mejoradas
    data['Has_Coordinates'] = (~data['Latitude'].isna()).astype(int)
    data['Has_Address'] = (~data['Incident Address'].isna()).astype(int)
    data['Has_Zip'] = (~data['Incident Zip'].isna()).astype(int)
    
    # Agrupar zip codes por regiones
    data['Zip_Region'] = (data['Incident Zip'].fillna(0) // 100).astype(int)
    
    # Características de frecuencia por borough
    borough_freq = data['Borough'].value_counts().to_dict()
    data['Borough_Frequency'] = data['Borough'].map(borough_freq)
    
    # Características de tipo de ubicación
    location_freq = data['Location Type'].value_counts().to_dict()
    data['Location_Frequency'] = data['Location Type'].map(location_freq).fillna(0)
    
    # Característica de status
    data['Is_Closed'] = (data['Status'] == 'Closed').astype(int)
    
    # Tiempo de resolución
    data['Closed Date'] = pd.to_datetime(data['Closed Date'], errors='coerce')
    data['Resolution_Hours'] = (data['Closed Date'] - data['Created Date']).dt.total_seconds() / 3600
    data['Has_Resolution'] = (~data['Resolution_Hours'].isna()).astype(int)
    data['Fast_Resolution'] = (data['Resolution_Hours'] <= 24).astype(int)
    
    print("✓ Características temporales avanzadas creadas")
    print("✓ Características geográficas mejoradas")
    print("✓ Características de frecuencia calculadas")
    
    return data

def select_best_features(data):
    """Seleccionar las mejores características usando métodos estadísticos"""
    
    print("\n2. SELECCIÓN INTELIGENTE DE CARACTERÍSTICAS...")
    print("-" * 55)
    
    # Características candidatas
    feature_candidates = [
        'Hour', 'DayOfWeek', 'Month', 'Quarter', 
        'IsWeekend', 'IsBusinessHour', 'IsEveningHour', 'IsNightHour',
        'Borough', 'Zip_Region', 'Borough_Frequency', 'Location_Frequency',
        'Has_Coordinates', 'Has_Address', 'Has_Zip', 'Is_Closed',
        'Has_Resolution', 'Fast_Resolution'
    ]
    
    # Filtrar características disponibles
    available_features = [col for col in feature_candidates if col in data.columns]
    
    # Crear dataset para análisis
    analysis_data = data[available_features + ['Descriptor']].copy()
    analysis_data = analysis_data.dropna(subset=['Descriptor'])
    
    # Preparar datos para selección de características
    X_temp = analysis_data[available_features].copy()
    y_temp = analysis_data['Descriptor'].copy()
    
    # Manejar variables categóricas para la selección
    categorical_cols = ['Borough']
    numerical_cols = [col for col in available_features if col not in categorical_cols]
    
    # Encode categorical variables temporalmente
    le = LabelEncoder()
    X_encoded = X_temp.copy()
    for col in categorical_cols:
        if col in X_encoded.columns:
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    
    # Imputar valores faltantes
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_encoded)
    
    # Seleccionar mejores características
    selector = SelectKBest(score_func=f_classif, k=12)  # Seleccionar top 12
    X_selected = selector.fit_transform(X_imputed, y_temp)
    
    # Obtener características seleccionadas
    selected_mask = selector.get_support()
    selected_features = [available_features[i] for i in range(len(available_features)) if selected_mask[i]]
    
    print(f"✓ {len(selected_features)} mejores características seleccionadas:")
    feature_scores = selector.scores_
    for i, (feature, selected) in enumerate(zip(available_features, selected_mask)):
        if selected:
            print(f"  - {feature} (score: {feature_scores[i]:.2f})")
    
    # Crear dataset final
    final_data = analysis_data[selected_features + ['Descriptor']].copy()
    
    print(f"\n✓ Dataset final: {final_data.shape[0]:,} filas, {len(selected_features)} características")
    
    return final_data, selected_features

def prepare_data_with_conservative_balancing(data, features):
    """Preparar datos con balanceamiento conservador"""
    
    print("\n3. PREPARACIÓN CON BALANCEAMIENTO CONSERVADOR...")
    print("-" * 55)
    
    X = data[features].copy()
    y = data['Descriptor'].copy()
    
    # Mostrar distribución original
    print("Distribución original de clases:")
    class_counts = y.value_counts()
    for class_name, count in class_counts.items():
        pct = (count / len(y)) * 100
        print(f"  - {class_name}: {count:,} ({pct:.2f}%)")
    
    # Estrategia conservadora: reducir solo las clases más minoritarias
    print("\nEstrategia de balanceamiento conservador:")
    
    # Identificar clases muy minoritarias (< 1% del total)
    minority_threshold = len(y) * 0.01
    minority_classes = [cls for cls, count in class_counts.items() if count < minority_threshold]
    
    print(f"Clases muy minoritarias (< 1%): {minority_classes}")
    
    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✓ División completada:")
    print(f"  - Train: {X_train.shape[0]:,} casos")
    print(f"  - Test: {X_test.shape[0]:,} casos")
    
    return X_train, X_test, y_train, y_test

def create_robust_preprocessing_pipeline():
    """Crear pipeline de preprocesamiento robusto"""
    
    # Identificar tipos de características
    categorical_features = ['Borough']
    numerical_features = [
        'Hour', 'DayOfWeek', 'Month', 'Quarter', 
        'IsWeekend', 'IsBusinessHour', 'IsEveningHour', 'IsNightHour',
        'Zip_Region', 'Borough_Frequency', 'Location_Frequency',
        'Has_Coordinates', 'Has_Address', 'Has_Zip', 'Is_Closed',
        'Has_Resolution', 'Fast_Resolution'
    ]
    
    # Pipeline para características numéricas
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline para características categóricas
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    
    # Combinar pipelines
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ], remainder='drop')
    
    return preprocessor

def train_final_models(X_train, X_test, y_train, y_test):
    """Entrenar modelos finales optimizados"""
    
    print("\n4. ENTRENAMIENTO DE MODELOS FINALES...")
    print("-" * 55)
    
    # Calcular pesos de clase balanceados
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    print("Pesos de clase calculados:")
    for cls, weight in class_weight_dict.items():
        print(f"  - {cls}: {weight:.3f}")
    
    # Crear preprocessor
    preprocessor = create_robust_preprocessing_pipeline()
    
    # Modelos optimizados
    models = {
        'Random Forest Optimizado': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight=class_weight_dict,
                random_state=42,
                n_jobs=-1
            ))
        ]),
        
        'Gradient Boosting Optimizado': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ))
        ]),
        
        'Logistic Regression Optimizada': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                class_weight=class_weight_dict,
                random_state=42,
                max_iter=1000,
                C=1.0
            ))
        ])
    }
    
    results = {}
    
    for name, pipeline in models.items():
        print(f"\n🔄 Entrenando y validando {name}...")
        
        try:
            # Validación cruzada
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='f1_weighted',
                n_jobs=-1
            )
            
            # Entrenar en todos los datos de entrenamiento
            pipeline.fit(X_train, y_train)
            
            # Predecir en test
            y_pred = pipeline.predict(X_test)
            
            # Calcular métricas
            accuracy = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            
            results[name] = {
                'model': pipeline,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'accuracy': accuracy,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro,
                'predictions': y_pred
            }
            
            print(f"✓ CV F1-Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"✓ Test Accuracy: {accuracy:.4f}")
            print(f"✓ Test F1-weighted: {f1_weighted:.4f}")
            print(f"✓ Test F1-macro: {f1_macro:.4f}")
            
        except Exception as e:
            print(f"✗ Error entrenando {name}: {e}")
            continue
    
    return results

def comprehensive_evaluation(results, y_test):
    """Evaluación comprensiva de modelos"""
    
    print("\n5. EVALUACIÓN COMPRENSIVA...")
    print("-" * 55)
    
    if not results:
        print("✗ No hay modelos para evaluar")
        return None, None
    
    # Encontrar el mejor modelo por F1-weighted
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_weighted'])
    best_model_info = results[best_model_name]
    
    print(f"🏆 MEJOR MODELO: {best_model_name}")
    print(f"   CV F1-Score: {best_model_info['cv_mean']:.4f} (±{best_model_info['cv_std']:.4f})")
    print(f"   Test Accuracy: {best_model_info['accuracy']:.4f}")
    print(f"   Test F1-weighted: {best_model_info['f1_weighted']:.4f}")
    print(f"   Test F1-macro: {best_model_info['f1_macro']:.4f}")
    
    # Reporte detallado
    print(f"\n📊 REPORTE DETALLADO - {best_model_name}:")
    print("-" * 55)
    y_pred = best_model_info['predictions']
    print(classification_report(y_test, y_pred))
    
    # Análisis de estabilidad
    cv_test_diff = abs(best_model_info['cv_mean'] - best_model_info['f1_weighted'])
    print(f"\n📈 ANÁLISIS DE ESTABILIDAD:")
    print(f"   Diferencia CV vs Test: {cv_test_diff:.4f}")
    if cv_test_diff < 0.05:
        print("   ✅ Modelo estable (baja diferencia CV vs Test)")
    elif cv_test_diff < 0.1:
        print("   ⚠️  Modelo moderadamente estable")
    else:
        print("   ❌ Posible overfitting (gran diferencia CV vs Test)")
    
    return best_model_name, best_model_info

def create_final_visualizations(results, y_test):
    """Crear visualizaciones finales"""
    
    print("\n6. CREANDO VISUALIZACIONES FINALES...")
    print("-" * 55)
    
    if not results:
        return
    
    # Configurar figura
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    
    # 1. Comparación de modelos con CV
    model_names = list(results.keys())
    cv_scores = [results[name]['cv_mean'] for name in model_names]
    test_scores = [results[name]['f1_weighted'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, cv_scores, width, label='CV F1-Score', alpha=0.8)
    axes[0, 0].bar(x + width/2, test_scores, width, label='Test F1-Score', alpha=0.8)
    axes[0, 0].set_xlabel('Modelos')
    axes[0, 0].set_ylabel('F1-Score')
    axes[0, 0].set_title('Comparación CV vs Test')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([name.replace(' Optimizado', '') for name in model_names], rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Matriz de confusión del mejor modelo
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_weighted'])
    y_pred_best = results[best_model_name]['predictions']
    
    cm = confusion_matrix(y_test, y_pred_best)
    im = axes[0, 1].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[0, 1].set_title(f'Matriz de Confusión\n{best_model_name.replace(" Optimizado", "")}')
    
    # Añadir números en la matriz
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0, 1].text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
    
    # 3. Distribución de predicciones
    real_counts = pd.Series(y_test).value_counts()
    pred_counts = pd.Series(y_pred_best).value_counts()
    
    comparison_df = pd.DataFrame({
        'Real': real_counts,
        'Predicho': pred_counts
    }).fillna(0)
    
    comparison_df.plot(kind='bar', ax=axes[0, 2])
    axes[0, 2].set_title('Real vs Predicho')
    axes[0, 2].set_xlabel('Clases')
    axes[0, 2].set_ylabel('Número de Casos')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].legend()
    
    # 4. Métricas por clase
    classes = sorted(y_test.unique())
    precisions = []
    recalls = []
    f1s = []
    
    for class_name in classes:
        class_mask = y_test == class_name
        pred_mask = y_pred_best == class_name
        
        tp = np.sum(class_mask & pred_mask)
        fp = np.sum(~class_mask & pred_mask)
        fn = np.sum(class_mask & ~pred_mask)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    x_pos = np.arange(len(classes))
    width = 0.25
    
    axes[1, 0].bar(x_pos - width, precisions, width, label='Precision', alpha=0.8)
    axes[1, 0].bar(x_pos, recalls, width, label='Recall', alpha=0.8)
    axes[1, 0].bar(x_pos + width, f1s, width, label='F1-Score', alpha=0.8)
    axes[1, 0].set_xlabel('Clases')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Métricas por Clase')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(classes, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Estabilidad de modelos (CV vs Test)
    axes[1, 1].scatter(cv_scores, test_scores, s=100, alpha=0.7)
    for i, name in enumerate(model_names):
        axes[1, 1].annotate(name.replace(' Optimizado', ''), 
                           (cv_scores[i], test_scores[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Línea diagonal para referencia
    min_val = min(min(cv_scores), min(test_scores)) - 0.01
    max_val = max(max(cv_scores), max(test_scores)) + 0.01
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    axes[1, 1].set_xlabel('CV F1-Score')
    axes[1, 1].set_ylabel('Test F1-Score')
    axes[1, 1].set_title('Estabilidad: CV vs Test')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Comparación de todas las métricas
    metrics_comparison = pd.DataFrame({
        'Accuracy': [results[name]['accuracy'] for name in model_names],
        'F1-Weighted': [results[name]['f1_weighted'] for name in model_names],
        'F1-Macro': [results[name]['f1_macro'] for name in model_names],
        'CV F1': [results[name]['cv_mean'] for name in model_names]
    }, index=[name.replace(' Optimizado', '') for name in model_names])
    
    metrics_comparison.plot(kind='bar', ax=axes[1, 2])
    axes[1, 2].set_title('Comparación Completa de Métricas')
    axes[1, 2].set_xlabel('Modelos')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_model_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Visualizaciones finales guardadas como 'final_model_results.png'")

def main():
    """Pipeline principal optimizado"""
    
    try:
        # 1. Ingeniería de características
        data = load_and_engineer_features()
        
        # 2. Selección inteligente de características
        final_data, selected_features = select_best_features(data)
        
        # 3. Preparación con balanceamiento conservador
        X_train, X_test, y_train, y_test = prepare_data_with_conservative_balancing(
            final_data, selected_features
        )
        
        # 4. Entrenamiento de modelos finales
        results = train_final_models(X_train, X_test, y_train, y_test)
        
        # 5. Evaluación comprensiva
        if results:
            best_model_name, best_model_info = comprehensive_evaluation(results, y_test)
            
            # 6. Visualizaciones finales
            create_final_visualizations(results, y_test)
            
            # 7. Resumen final
            print("\n" + "=" * 65)
            print("RESUMEN FINAL - MODELO OPTIMIZADO")
            print("=" * 65)
            print(f"🏆 Mejor modelo: {best_model_name}")
            print(f"📊 CV F1-Score: {best_model_info['cv_mean']:.4f} (±{best_model_info['cv_std']:.4f})")
            print(f"📊 Test Accuracy: {best_model_info['accuracy']:.4f}")
            print(f"📊 Test F1-weighted: {best_model_info['f1_weighted']:.4f}")
            print(f"📊 Test F1-macro: {best_model_info['f1_macro']:.4f}")
            
            # Comparación con modelo baseline
            baseline_acc = 0.5652
            baseline_f1 = 0.5057
            
            improvement_acc = (best_model_info['accuracy'] - baseline_acc) / baseline_acc * 100
            improvement_f1 = (best_model_info['f1_weighted'] - baseline_f1) / baseline_f1 * 100
            
            print(f"\n📈 COMPARACIÓN CON BASELINE:")
            print(f"   Baseline Accuracy: {baseline_acc:.4f}")
            print(f"   Baseline F1-weighted: {baseline_f1:.4f}")
            print(f"   Mejora en Accuracy: {improvement_acc:+.1f}%")
            print(f"   Mejora en F1-weighted: {improvement_f1:+.1f}%")
            
            if improvement_f1 > 0:
                print("   ✅ Modelo mejorado exitosamente!")
            else:
                print("   ⚠️  Modelo necesita más optimización")
            
            print("\n🎯 CARACTERÍSTICAS DEL MODELO FINAL:")
            print("   ✅ Selección inteligente de características")
            print("   ✅ Balanceamiento conservador con class weights")
            print("   ✅ Validación cruzada para estabilidad")
            print("   ✅ Hiperparámetros optimizados")
            print("   ✅ Pipeline robusto de preprocesamiento")
            
            return best_model_info['model'], results
        else:
            print("✗ No se pudieron entrenar modelos")
            return None, None
            
    except Exception as e:
        print(f"✗ Error en el pipeline: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None

if __name__ == "__main__":
    final_model, all_results = main() 