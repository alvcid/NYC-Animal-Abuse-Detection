import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Para balanceamiento de clases
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter

def load_and_clean_data():
    """Cargar y limpiar datos con manejo robusto de valores faltantes"""
    
    print("=" * 60)
    print("MODELO MEJORADO - NYC ANIMAL ABUSE PREDICTION")
    print("=" * 60)
    
    print("\n1. CARGANDO Y LIMPIANDO DATOS...")
    print("-" * 40)
    
    # Cargar datos
    df = pd.read_csv('Animal_Abuse.csv/Animal_Abuse.csv')
    print(f"✓ Dataset original: {df.shape[0]:,} filas, {df.shape[1]} columnas")
    
    # Crear características mejoradas
    data = df.copy()
    
    # Características temporales
    data['Created Date'] = pd.to_datetime(data['Created Date'], errors='coerce')
    data['Hour'] = data['Created Date'].dt.hour
    data['DayOfWeek'] = data['Created Date'].dt.dayofweek
    data['Month'] = data['Created Date'].dt.month
    data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Características geográficas
    data['Has_Coordinates'] = (~data['Latitude'].isna()).astype(int)
    data['Has_Address'] = (~data['Incident Address'].isna()).astype(int)
    
    # Seleccionar características más robustas (sin valores faltantes complejos)
    features = [
        'Hour', 'DayOfWeek', 'Month', 'IsWeekend',
        'Borough', 'Has_Coordinates', 'Has_Address'
    ]
    
    # Crear dataset limpio
    clean_data = data[features + ['Descriptor']].copy()
    
    # Eliminar filas con valores faltantes en características críticas
    clean_data = clean_data.dropna(subset=['Descriptor'])
    
    print(f"✓ Dataset limpio: {clean_data.shape[0]:,} filas, {len(features)} características")
    
    return clean_data, features

def prepare_data_for_modeling(data, features):
    """Preparar datos para modelado"""
    
    print("\n2. PREPARANDO DATOS PARA MODELADO...")
    print("-" * 40)
    
    X = data[features].copy()
    y = data['Descriptor'].copy()
    
    # Mostrar distribución de clases
    print("Distribución de clases:")
    class_counts = y.value_counts()
    for class_name, count in class_counts.items():
        pct = (count / len(y)) * 100
        print(f"  - {class_name}: {count:,} ({pct:.2f}%)")
    
    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✓ División completada:")
    print(f"  - Train: {X_train.shape[0]:,} casos")
    print(f"  - Test: {X_test.shape[0]:,} casos")
    
    return X_train, X_test, y_train, y_test

def create_preprocessing_pipeline():
    """Crear pipeline de preprocesamiento robusto"""
    
    # Identificar tipos de características
    categorical_features = ['Borough']
    numerical_features = ['Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'Has_Coordinates', 'Has_Address']
    
    # Pipelines específicos
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
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
    
    return preprocessor

def train_improved_models(X_train, X_test, y_train, y_test):
    """Entrenar modelos mejorados"""
    
    print("\n3. ENTRENANDO MODELOS MEJORADOS...")
    print("-" * 40)
    
    # Crear preprocessor
    preprocessor = create_preprocessing_pipeline()
    
    # Modelos a probar
    models = {
        'Random Forest Balanceado': RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=42, 
            class_weight='balanced', n_jobs=-1
        ),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(
            max_iter=200, max_depth=10, random_state=42,
            class_weight='balanced'
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42, max_iter=1000, class_weight='balanced'
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔄 Entrenando {name}...")
        
        try:
            # Crear pipeline con balanceamiento
            pipeline = ImbPipeline([
                ('preprocessor', preprocessor),
                ('balancer', RandomOverSampler(random_state=42)),
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

def evaluate_models(results, y_test):
    """Evaluar y comparar modelos"""
    
    print("\n4. EVALUACIÓN DE MODELOS...")
    print("-" * 40)
    
    if not results:
        print("✗ No hay modelos para evaluar")
        return None, None
    
    # Encontrar el mejor modelo
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_weighted'])
    best_model_info = results[best_model_name]
    
    print(f"🏆 MEJOR MODELO: {best_model_name}")
    print(f"   Accuracy: {best_model_info['accuracy']:.4f}")
    print(f"   F1-Score (weighted): {best_model_info['f1_weighted']:.4f}")
    print(f"   F1-Score (macro): {best_model_info['f1_macro']:.4f}")
    
    # Reporte detallado
    print(f"\n📊 REPORTE DETALLADO:")
    print("-" * 40)
    y_pred = best_model_info['predictions']
    print(classification_report(y_test, y_pred))
    
    return best_model_name, best_model_info

def create_visualizations(results, y_test):
    """Crear visualizaciones de resultados"""
    
    print("\n5. CREANDO VISUALIZACIONES...")
    print("-" * 40)
    
    if not results:
        return
    
    # Configurar figura
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Comparación de modelos
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    f1_scores = [results[name]['f1_weighted'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    axes[0, 0].bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
    axes[0, 0].set_xlabel('Modelos')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Comparación de Modelos')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Matriz de confusión del mejor modelo
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_weighted'])
    y_pred_best = results[best_model_name]['predictions']
    
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 1], cmap='Blues')
    axes[0, 1].set_title(f'Matriz de Confusión - {best_model_name}')
    axes[0, 1].set_xlabel('Predicción')
    axes[0, 1].set_ylabel('Real')
    
    # 3. Distribución de predicciones
    pred_counts = pd.Series(y_pred_best).value_counts()
    axes[0, 2].pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%')
    axes[0, 2].set_title('Distribución de Predicciones')
    
    # 4. F1-Score por clase
    classes = sorted(y_test.unique())
    f1_per_class = []
    for class_name in classes:
        f1 = f1_score(y_test == class_name, y_pred_best == class_name)
        f1_per_class.append(f1)
    
    axes[1, 0].barh(classes, f1_per_class)
    axes[1, 0].set_xlabel('F1-Score')
    axes[1, 0].set_title('F1-Score por Clase')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Comparación de todas las métricas
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
    
    metrics_df.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('Comparación de Todas las Métricas')
    axes[1, 1].set_xlabel('Modelos')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Distribución de errores
    errors = y_test != y_pred_best
    error_by_class = pd.Series(y_test[errors]).value_counts()
    axes[1, 2].bar(error_by_class.index, error_by_class.values, color='red', alpha=0.7)
    axes[1, 2].set_title('Errores por Clase Real')
    axes[1, 2].set_xlabel('Clase Real')
    axes[1, 2].set_ylabel('Número de Errores')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('improved_model_v2_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Visualizaciones guardadas como 'improved_model_v2_results.png'")

def main():
    """Función principal"""
    
    try:
        # 1. Cargar y limpiar datos
        data, features = load_and_clean_data()
        
        # 2. Preparar datos
        X_train, X_test, y_train, y_test = prepare_data_for_modeling(data, features)
        
        # 3. Entrenar modelos
        results = train_improved_models(X_train, X_test, y_train, y_test)
        
        # 4. Evaluar modelos
        if results:
            best_model_name, best_model_info = evaluate_models(results, y_test)
            
            # 5. Crear visualizaciones
            create_visualizations(results, y_test)
            
            # 6. Resumen final
            print("\n" + "=" * 60)
            print("RESUMEN FINAL DEL MODELO MEJORADO")
            print("=" * 60)
            print(f"🏆 Mejor modelo: {best_model_name}")
            print(f"📊 Accuracy: {best_model_info['accuracy']:.4f}")
            print(f"📊 F1-Score (weighted): {best_model_info['f1_weighted']:.4f}")
            print(f"📊 F1-Score (macro): {best_model_info['f1_macro']:.4f}")
            
            # Comparar con modelo anterior
            prev_accuracy = 0.5652
            prev_f1 = 0.5057
            
            improvement_acc = (best_model_info['accuracy'] - prev_accuracy) / prev_accuracy * 100
            improvement_f1 = (best_model_info['f1_weighted'] - prev_f1) / prev_f1 * 100
            
            print(f"\n📈 MEJORAS:")
            print(f"   Accuracy: {improvement_acc:+.1f}% de mejora")
            print(f"   F1-Score: {improvement_f1:+.1f}% de mejora")
            
            print("\n✅ Modelo mejorado completado exitosamente!")
            
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
    best_model, all_results = main() 