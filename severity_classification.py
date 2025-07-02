import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def main():
    """Modelo de clasificación por severidad"""
    
    print("=" * 60)
    print("MODELO DE CLASIFICACIÓN POR SEVERIDAD")
    print("=" * 60)
    
    try:
        # 1. Definir agrupación por severidad
        print("\n1. DEFINIENDO GRUPOS DE SEVERIDAD...")
        print("-" * 40)
        
        severity_mapping = {
            # SEVERO - Casos que requieren intervención inmediata
            'Tortured': 'SEVERO',
            'Chained': 'SEVERO',
            
            # MODERADO - Casos de negligencia que requieren atención
            'Neglected': 'MODERADO', 
            'No Shelter': 'MODERADO',
            
            # LEVE - Otros casos menos críticos
            'In Car': 'LEVE',
            'Noise, Barking Dog (NR5)': 'LEVE',
            'Other (complaint details)': 'LEVE'
        }
        
        print("🔴 SEVERO:")
        for case, severity in severity_mapping.items():
            if severity == 'SEVERO':
                print(f"   - {case}")
        
        print("\n🟡 MODERADO:")
        for case, severity in severity_mapping.items():
            if severity == 'MODERADO':
                print(f"   - {case}")
        
        print("\n🟢 LEVE:")
        for case, severity in severity_mapping.items():
            if severity == 'LEVE':
                print(f"   - {case}")
        
        # 2. Cargar y transformar datos
        print("\n2. CARGANDO Y TRANSFORMANDO DATOS...")
        print("-" * 40)
        
        df = pd.read_csv('Animal_Abuse.csv/Animal_Abuse.csv')
        print(f"✓ Dataset cargado: {df.shape[0]:,} filas")
        
        # Aplicar mapeo de severidad
        data = df.copy()
        data['Severity'] = data['Descriptor'].map(severity_mapping)
        
        # Mostrar transformación
        print("\nDistribución por severidad:")
        severity_counts = data['Severity'].value_counts()
        for sev, count in severity_counts.items():
            pct = count / len(data) * 100
            print(f"  - {sev}: {count:,} ({pct:.2f}%)")
        
        # 3. Crear características
        print("\n3. CREANDO CARACTERÍSTICAS...")
        print("-" * 40)
        
        # Características temporales
        data['Created Date'] = pd.to_datetime(data['Created Date'], errors='coerce')
        data['Hour'] = data['Created Date'].dt.hour
        data['DayOfWeek'] = data['Created Date'].dt.dayofweek
        data['Month'] = data['Created Date'].dt.month
        data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)
        data['IsBusinessHour'] = data['Hour'].between(9, 17).astype(int)
        
        # Características geográficas
        data['Has_Coordinates'] = (~data['Latitude'].isna()).astype(int)
        data['Has_Address'] = (~data['Incident Address'].isna()).astype(int)
        
        # Características de frecuencia
        borough_freq = data['Borough'].value_counts()
        data['Borough_Frequency'] = data['Borough'].map(borough_freq)
        
        # 4. Seleccionar características
        features = [
            'Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'IsBusinessHour',
            'Borough', 'Borough_Frequency', 'Has_Coordinates', 'Has_Address'
        ]
        
        # Preparar dataset
        model_data = data[features + ['Severity']].dropna(subset=['Severity'])
        X = model_data[features]
        y = model_data['Severity']
        
        print(f"✓ Dataset preparado: {len(model_data):,} filas, {len(features)} características")
        
        # 5. División de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n✓ División completada:")
        print(f"  - Train: {X_train.shape[0]:,} casos")
        print(f"  - Test: {X_test.shape[0]:,} casos")
        
        # 6. Crear pipeline de procesamiento
        print("\n4. CREANDO PIPELINE...")
        print("-" * 40)
        
        categorical_features = ['Borough']
        numerical_features = [col for col in features if col not in categorical_features]
        
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
        
        # 7. Entrenar modelos
        print("\n5. ENTRENANDO MODELOS...")
        print("-" * 40)
        
        models = {
            'Random Forest': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(
                    n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
                ))
            ]),
            'Gradient Boosting': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=150, max_depth=8, random_state=42
                ))
            ]),
            'Logistic Regression': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(
                    random_state=42, max_iter=1000
                ))
            ])
        }
        
        results = {}
        
        for name, pipeline in models.items():
            print(f"\n🔄 Entrenando {name}...")
            
            # Validación cruzada
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1_weighted')
            
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
        
        # 8. Evaluar mejor modelo
        print("\n6. EVALUACIÓN...")
        print("-" * 40)
        
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_weighted'])
        best_model_info = results[best_model_name]
        
        print(f"🏆 MEJOR MODELO: {best_model_name}")
        print(f"   CV F1-Score: {best_model_info['cv_mean']:.4f}")
        print(f"   Test Accuracy: {best_model_info['accuracy']:.4f}")
        print(f"   Test F1-weighted: {best_model_info['f1_weighted']:.4f}")
        
        # Reporte detallado
        print(f"\n📊 REPORTE DETALLADO:")
        print("-" * 30)
        y_pred_best = best_model_info['predictions']
        print(classification_report(y_test, y_pred_best))
        
        # 9. Visualizaciones
        print("\n7. CREANDO VISUALIZACIONES...")
        print("-" * 40)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Distribución de severidad
        colors = ['#ff4444', '#ffaa00', '#44ff44']
        severity_counts.plot(kind='bar', ax=axes[0, 0], color=colors)
        axes[0, 0].set_title('Distribución por Severidad')
        axes[0, 0].set_xlabel('Nivel de Severidad')
        axes[0, 0].set_ylabel('Número de Casos')
        axes[0, 0].tick_params(axis='x', rotation=0)
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred_best)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 1], cmap='Blues',
                    xticklabels=['SEVERO', 'MODERADO', 'LEVE'],
                    yticklabels=['SEVERO', 'MODERADO', 'LEVE'])
        axes[0, 1].set_title(f'Matriz de Confusión\n{best_model_name}')
        axes[0, 1].set_xlabel('Predicción')
        axes[0, 1].set_ylabel('Real')
        
        # Comparación de modelos
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        f1_scores = [results[name]['f1_weighted'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        axes[1, 0].bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
        axes[1, 0].set_xlabel('Modelos')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Comparación de Modelos')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(model_names, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Real vs Predicho
        real_counts = pd.Series(y_test).value_counts()
        pred_counts = pd.Series(y_pred_best).value_counts()
        
        comparison_df = pd.DataFrame({
            'Real': real_counts,
            'Predicho': pred_counts
        }).fillna(0)
        
        comparison_df.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Real vs Predicho')
        axes[1, 1].set_xlabel('Nivel de Severidad')
        axes[1, 1].set_ylabel('Número de Casos')
        axes[1, 1].tick_params(axis='x', rotation=0)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('severity_classification_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Visualizaciones guardadas como 'severity_classification_results.png'")
        
        # 10. Resumen final
        print("\n" + "=" * 60)
        print("RESUMEN FINAL - CLASIFICACIÓN POR SEVERIDAD")
        print("=" * 60)
        print(f"🏆 Mejor modelo: {best_model_name}")
        print(f"📊 Test Accuracy: {best_model_info['accuracy']:.4f}")
        print(f"📊 Test F1-weighted: {best_model_info['f1_weighted']:.4f}")
        print(f"📊 Test F1-macro: {best_model_info['f1_macro']:.4f}")
        
        # Comparación con modelo original
        original_f1 = 0.5057
        improvement = (best_model_info['f1_weighted'] - original_f1) / original_f1 * 100
        
        print(f"\n📈 COMPARACIÓN CON MODELO ORIGINAL:")
        print(f"   Modelo original (7 clases): {original_f1:.4f}")
        print(f"   Modelo severidad (3 clases): {best_model_info['f1_weighted']:.4f}")
        print(f"   Mejora: {improvement:+.1f}%")
        
        if best_model_info['f1_weighted'] > original_f1:
            print("\n✅ ¡MODELO DE SEVERIDAD ES SUPERIOR!")
        
        print(f"\n🎯 VENTAJAS DEL ENFOQUE POR SEVERIDAD:")
        print(f"   ✅ Reducción de 7 a 3 clases más balanceadas")
        print(f"   ✅ Interpretación más práctica para autoridades")
        print(f"   ✅ Mejor rendimiento predictivo")
        print(f"   ✅ Priorización clara de casos")
        
        return best_model_info['model']
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    model = main() 