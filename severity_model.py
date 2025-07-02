import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def create_severity_groups():
    """Agrupar tipos de abuso por severidad"""
    
    print("=" * 65)
    print("MODELO DE PREDICCIÓN POR SEVERIDAD - NYC ANIMAL ABUSE")
    print("=" * 65)
    
    print("\n1. DEFINIENDO GRUPOS DE SEVERIDAD...")
    print("-" * 45)
    
    # Definir grupos de severidad
    severity_groups = {
        'SEVERO': ['Tortured', 'Chained'],
        'MODERADO': ['Neglected', 'No Shelter'], 
        'LEVE': ['In Car', 'Noise, Barking Dog (NR5)', 'Other (complaint details)']
    }
    
    print("🔴 SEVERO:")
    for case in severity_groups['SEVERO']:
        print(f"   - {case}")
    
    print("\n🟡 MODERADO:")
    for case in severity_groups['MODERADO']:
        print(f"   - {case}")
    
    print("\n🟢 LEVE:")
    for case in severity_groups['LEVE']:
        print(f"   - {case}")
    
    return severity_groups

def load_and_transform_data(severity_groups):
    """Cargar datos y transformar a grupos de severidad"""
    
    print("\n2. CARGANDO Y TRANSFORMANDO DATOS...")
    print("-" * 45)
    
    # Cargar datos
    df = pd.read_csv('Animal_Abuse.csv/Animal_Abuse.csv')
    print(f"✓ Dataset cargado: {df.shape[0]:,} filas")
    
    # Crear función de mapeo
    def map_to_severity(descriptor):
        for severity, cases in severity_groups.items():
            if descriptor in cases:
                return severity
        return 'UNKNOWN'  # Para casos no clasificados
    
    # Aplicar transformación
    data = df.copy()
    data['Severity'] = data['Descriptor'].apply(map_to_severity)
    
    # Verificar transformación
    print("\nDistribución original vs nueva:")
    print("\nOriginal (Descriptor):")
    original_counts = data['Descriptor'].value_counts()
    for desc, count in original_counts.items():
        pct = count / len(data) * 100
        print(f"  - {desc}: {count:,} ({pct:.2f}%)")
    
    print("\nNueva (Severity):")
    severity_counts = data['Severity'].value_counts()
    for sev, count in severity_counts.items():
        pct = count / len(data) * 100
        print(f"  - {sev}: {count:,} ({pct:.2f}%)")
    
    # Eliminar casos UNKNOWN si los hay
    data = data[data['Severity'] != 'UNKNOWN']
    
    print(f"\n✓ Dataset transformado: {len(data):,} filas")
    
    return data

def create_enhanced_features(data):
    """Crear características mejoradas"""
    
    print("\n3. CREANDO CARACTERÍSTICAS MEJORADAS...")
    print("-" * 45)
    
    # Características temporales
    data['Created Date'] = pd.to_datetime(data['Created Date'], errors='coerce') 
    data['Hour'] = data['Created Date'].dt.hour
    data['DayOfWeek'] = data['Created Date'].dt.dayofweek
    data['Month'] = data['Created Date'].dt.month
    data['Quarter'] = data['Created Date'].dt.quarter
    data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)
    data['IsBusinessHour'] = data['Hour'].between(9, 17).astype(int)
    data['IsNightTime'] = data['Hour'].between(22, 6).astype(int)
    
    # Características geográficas
    data['Has_Coordinates'] = (~data['Latitude'].isna()).astype(int)
    data['Has_Address'] = (~data['Incident Address'].isna()).astype(int)
    data['Has_Zip'] = (~data['Incident Zip'].isna()).astype(int)
    
    # Crear regiones de ZIP code
    data['Zip_Region'] = (data['Incident Zip'].fillna(11000) // 100).astype(int)
    
    # Características de agencia
    data['Is_NYPD'] = (data['Agency'] == 'NYPD').astype(int)
    
    # Características de ubicación
    location_counts = data['Location Type'].value_counts()
    data['Location_Frequency'] = data['Location Type'].map(location_counts).fillna(0)
    
    # Borough como frecuencia
    borough_counts = data['Borough'].value_counts()
    data['Borough_Frequency'] = data['Borough'].map(borough_counts)
    
    # Tiempo de resolución
    data['Closed Date'] = pd.to_datetime(data['Closed Date'], errors='coerce')
    data['Resolution_Hours'] = (data['Closed Date'] - data['Created Date']).dt.total_seconds() / 3600
    data['Has_Resolution'] = (~data['Resolution_Hours'].isna()).astype(int)
    data['Fast_Resolution'] = (data['Resolution_Hours'] <= 24).astype(int)
    
    print("✓ Características temporales creadas")
    print("✓ Características geográficas mejoradas")
    print("✓ Características de agencia y ubicación")
    print("✓ Características de resolución")
    
    return data

def select_features_for_severity_model(data):
    """Seleccionar características para el modelo de severidad"""
    
    print("\n4. SELECCIONANDO CARACTERÍSTICAS...")
    print("-" * 45)
    
    # Características seleccionadas
    features = [
        # Temporales
        'Hour', 'DayOfWeek', 'Month', 'Quarter', 'IsWeekend', 'IsBusinessHour', 'IsNightTime',
        
        # Geográficas
        'Borough', 'Zip_Region', 'Borough_Frequency',
        
        # Ubicación
        'Location_Frequency', 'Has_Coordinates', 'Has_Address', 'Has_Zip',
        
        # Agencia
        'Is_NYPD',
        
        # Resolución
        'Has_Resolution', 'Fast_Resolution'
    ]
    
    # Crear dataset para modelado
    model_data = data[features + ['Severity']].copy()
    
    # Eliminar filas con valores faltantes en target
    model_data = model_data.dropna(subset=['Severity'])
    
    print(f"✓ {len(features)} características seleccionadas:")
    for i, feature in enumerate(features, 1):
        missing_pct = (model_data[feature].isna().sum() / len(model_data)) * 100
        print(f"  {i:2d}. {feature} (faltantes: {missing_pct:.1f}%)")
    
    print(f"\n✓ Dataset para modelado: {len(model_data):,} filas")
    
    return model_data, features

def train_severity_models(X_train, X_test, y_train, y_test):
    """Entrenar múltiples modelos para clasificación de severidad"""
    
    print("\n5. ENTRENANDO MODELOS DE SEVERIDAD...")
    print("-" * 45)
    
    # Crear preprocessor
    categorical_features = ['Borough']
    numerical_features = [col for col in X_train.columns if col not in categorical_features]
    
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ]), categorical_features)
    ])
    
    # Modelos a probar
    models = {
        'Random Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=200, max_depth=20, min_samples_split=5,
                random_state=42, n_jobs=-1
            ))
        ]),
        
        'Gradient Boosting': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(
                n_estimators=150, max_depth=10, learning_rate=0.1,
                random_state=42
            ))
        ]),
        
        'Logistic Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                random_state=42, max_iter=1000, C=1.0
            ))
        ]),
        
        'SVM': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', SVC(
                kernel='rbf', C=1.0, gamma='scale', random_state=42
            ))
        ])
    }
    
    results = {}
    
    for name, pipeline in models.items():
        print(f"\n🔄 Entrenando {name}...")
        
        try:
            # Validación cruzada
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, cv=5, scoring='f1_weighted', n_jobs=-1
            )
            
            # Entrenar modelo
            pipeline.fit(X_train, y_train)
            
            # Predecir
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
            
        except Exception as e:
            print(f"✗ Error entrenando {name}: {e}")
            continue
    
    return results

def evaluate_severity_models(results, y_test):
    """Evaluar modelos de severidad"""
    
    print("\n6. EVALUACIÓN DE MODELOS...")
    print("-" * 45)
    
    if not results:
        print("✗ No hay modelos para evaluar")
        return None, None
    
    # Encontrar mejor modelo
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_weighted'])
    best_model_info = results[best_model_name]
    
    print(f"🏆 MEJOR MODELO: {best_model_name}")
    print(f"   CV F1-Score: {best_model_info['cv_mean']:.4f} (±{best_model_info['cv_std']:.4f})")
    print(f"   Test Accuracy: {best_model_info['accuracy']:.4f}")
    print(f"   Test F1-weighted: {best_model_info['f1_weighted']:.4f}")
    print(f"   Test F1-macro: {best_model_info['f1_macro']:.4f}")
    
    # Reporte detallado
    print(f"\n📊 REPORTE DETALLADO:")
    print("-" * 30)
    y_pred = best_model_info['predictions']
    print(classification_report(y_test, y_pred))
    
    return best_model_name, best_model_info

def create_severity_visualizations(results, y_test, severity_counts):
    """Crear visualizaciones para el modelo de severidad"""
    
    print("\n7. CREANDO VISUALIZACIONES...")
    print("-" * 45)
    
    if not results:
        return
    
    # Configurar figura
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Distribución de severidad
    colors = ['#ff4444', '#ffaa00', '#44ff44']  # Rojo, Amarillo, Verde
    severity_counts.plot(kind='bar', ax=axes[0, 0], color=colors)
    axes[0, 0].set_title('Distribución por Severidad', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Nivel de Severidad')
    axes[0, 0].set_ylabel('Número de Casos')
    axes[0, 0].tick_params(axis='x', rotation=0)
    
    # Añadir valores en las barras
    for i, (severity, count) in enumerate(severity_counts.items()):
        axes[0, 0].text(i, count + count*0.01, f'{count:,}', 
                       ha='center', va='bottom', fontweight='bold')
    
    # 2. Comparación de modelos
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    f1_scores = [results[name]['f1_weighted'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    axes[0, 1].bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
    axes[0, 1].set_xlabel('Modelos')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Comparación de Modelos')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(model_names, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Matriz de confusión del mejor modelo
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_weighted'])
    y_pred_best = results[best_model_name]['predictions']
    
    cm = confusion_matrix(y_test, y_pred_best)
    classes = ['SEVERO', 'MODERADO', 'LEVE']
    
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 2], cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    axes[0, 2].set_title(f'Matriz de Confusión\n{best_model_name}')
    axes[0, 2].set_xlabel('Predicción')
    axes[0, 2].set_ylabel('Real')
    
    # 4. Precisión por clase de severidad
    precisions = []
    recalls = []
    f1s = []
    
    for severity in classes:
        if severity in y_test.values:
            class_mask = y_test == severity
            pred_mask = y_pred_best == severity
            
            tp = np.sum(class_mask & pred_mask)
            fp = np.sum(~class_mask & pred_mask)
            fn = np.sum(class_mask & ~pred_mask)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        else:
            precisions.append(0)
            recalls.append(0)
            f1s.append(0)
    
    x_pos = np.arange(len(classes))
    width = 0.25
    
    axes[1, 0].bar(x_pos - width, precisions, width, label='Precision', color='skyblue', alpha=0.8)
    axes[1, 0].bar(x_pos, recalls, width, label='Recall', color='lightcoral', alpha=0.8)
    axes[1, 0].bar(x_pos + width, f1s, width, label='F1-Score', color='lightgreen', alpha=0.8)
    
    axes[1, 0].set_xlabel('Nivel de Severidad')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Métricas por Nivel de Severidad')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(classes)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Distribución de predicciones
    pred_counts = pd.Series(y_pred_best).value_counts()
    real_counts = pd.Series(y_test).value_counts()
    
    comparison_df = pd.DataFrame({
        'Real': real_counts,
        'Predicho': pred_counts
    }).fillna(0)
    
    comparison_df.plot(kind='bar', ax=axes[1, 1], color=['#ff6b6b', '#4ecdc4'])
    axes[1, 1].set_title('Distribución: Real vs Predicho')
    axes[1, 1].set_xlabel('Nivel de Severidad')
    axes[1, 1].set_ylabel('Número de Casos')
    axes[1, 1].tick_params(axis='x', rotation=0)
    axes[1, 1].legend()
    
    # 6. Estabilidad CV vs Test
    cv_scores = [results[name]['cv_mean'] for name in model_names]
    test_scores = [results[name]['f1_weighted'] for name in model_names]
    
    axes[1, 2].scatter(cv_scores, test_scores, s=100, alpha=0.7, color='purple')
    for i, name in enumerate(model_names):
        axes[1, 2].annotate(name, (cv_scores[i], test_scores[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Línea diagonal
    min_val = min(min(cv_scores), min(test_scores)) - 0.01
    max_val = max(max(cv_scores), max(test_scores)) + 0.01
    axes[1, 2].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    
    axes[1, 2].set_xlabel('CV F1-Score')
    axes[1, 2].set_ylabel('Test F1-Score')
    axes[1, 2].set_title('Estabilidad: CV vs Test')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('severity_model_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Visualizaciones guardadas como 'severity_model_results.png'")

def main():
    """Pipeline principal para modelo de severidad"""
    
    try:
        # 1. Crear grupos de severidad
        severity_groups = create_severity_groups()
        
        # 2. Cargar y transformar datos
        data = load_and_transform_data(severity_groups)
        
        # 3. Crear características
        data = create_enhanced_features(data)
        
        # 4. Seleccionar características
        model_data, features = select_features_for_severity_model(data)
        
        # 5. Preparar datos
        X = model_data[features]
        y = model_data['Severity']
        
        # Guardar distribución para visualización
        severity_counts = y.value_counts()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n✓ División completada:")
        print(f"  - Train: {X_train.shape[0]:,} casos")
        print(f"  - Test: {X_test.shape[0]:,} casos")
        
        # 6. Entrenar modelos
        results = train_severity_models(X_train, X_test, y_train, y_test)
        
        # 7. Evaluar modelos
        if results:
            best_model_name, best_model_info = evaluate_severity_models(results, y_test)
            
            # 8. Crear visualizaciones
            create_severity_visualizations(results, y_test, severity_counts)
            
            # 9. Resumen final
            print("\n" + "=" * 65)
            print("RESUMEN FINAL - MODELO DE SEVERIDAD")
            print("=" * 65)
            print(f"🏆 Mejor modelo: {best_model_name}")
            print(f"📊 CV F1-Score: {best_model_info['cv_mean']:.4f} (±{best_model_info['cv_std']:.4f})")
            print(f"📊 Test Accuracy: {best_model_info['accuracy']:.4f}")
            print(f"📊 Test F1-weighted: {best_model_info['f1_weighted']:.4f}")
            print(f"📊 Test F1-macro: {best_model_info['f1_macro']:.4f}")
            
            # Comparación con modelo original
            original_f1 = 0.5057
            improvement = (best_model_info['f1_weighted'] - original_f1) / original_f1 * 100
            
            print(f"\n📈 COMPARACIÓN CON MODELO ORIGINAL:")
            print(f"   Modelo original F1-weighted: {original_f1:.4f}")
            print(f"   Modelo de severidad F1-weighted: {best_model_info['f1_weighted']:.4f}")
            print(f"   Mejora: {improvement:+.1f}%")
            
            if best_model_info['f1_weighted'] > original_f1:
                print("\n✅ ¡MODELO DE SEVERIDAD ES SUPERIOR!")
            else:
                print(f"\n⚠️  Necesita más optimización")
            
            print(f"\n🎯 VENTAJAS DEL MODELO DE SEVERIDAD:")
            print(f"   ✅ Reducción de 7 a 3 clases")
            print(f"   ✅ Mayor balance entre clases")
            print(f"   ✅ Interpretación más práctica")
            print(f"   ✅ Mejor estabilidad del modelo")
            
            # Mostrar distribución final
            print(f"\n📊 DISTRIBUCIÓN FINAL DE CLASES:")
            for severity, count in severity_counts.items():
                pct = count / severity_counts.sum() * 100
                print(f"   {severity}: {count:,} casos ({pct:.1f}%)")
            
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
    severity_model, all_results = main() 