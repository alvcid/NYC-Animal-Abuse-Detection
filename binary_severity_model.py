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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def main():
    """Modelo binario de clasificación CRITICO vs NO_CRITICO"""
    
    print("=" * 65)
    print("MODELO BINARIO: CASOS CRÍTICOS vs NO CRÍTICOS")
    print("=" * 65)
    
    try:
        # 1. Definir clasificación binaria
        print("\n1. DEFINIENDO CLASIFICACIÓN BINARIA...")
        print("-" * 45)
        
        # Clasificación binaria más simple y práctica
        binary_mapping = {
            # CRÍTICO - Requiere intervención inmediata
            'Tortured': 'CRITICO',
            'Chained': 'CRITICO',
            
            # NO CRÍTICO - Otros casos 
            'Neglected': 'NO_CRITICO',
            'No Shelter': 'NO_CRITICO',
            'In Car': 'NO_CRITICO',
            'Noise, Barking Dog (NR5)': 'NO_CRITICO',
            'Other (complaint details)': 'NO_CRITICO'
        }
        
        print("🔴 CRÍTICO (Intervención inmediata):")
        for case, category in binary_mapping.items():
            if category == 'CRITICO':
                print(f"   - {case}")
        
        print("\n🟢 NO CRÍTICO (Otros casos):")
        for case, category in binary_mapping.items():
            if category == 'NO_CRITICO':
                print(f"   - {case}")
        
        # 2. Cargar y transformar datos
        print("\n2. CARGANDO Y TRANSFORMANDO DATOS...")
        print("-" * 45)
        
        df = pd.read_csv('Animal_Abuse.csv/Animal_Abuse.csv')
        print(f"✓ Dataset cargado: {df.shape[0]:,} filas")
        
        # Aplicar clasificación binaria
        data = df.copy()
        data['Priority'] = data['Descriptor'].map(binary_mapping)
        
        # Mostrar distribución binaria
        print("\nDistribución binaria:")
        priority_counts = data['Priority'].value_counts()
        for priority, count in priority_counts.items():
            pct = count / len(data) * 100
            print(f"  - {priority}: {count:,} ({pct:.2f}%)")
        
        # 3. Crear características avanzadas
        print("\n3. CREANDO CARACTERÍSTICAS AVANZADAS...")
        print("-" * 45)
        
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
        
        # Crear regiones más específicas
        data['Zip_Region'] = (data['Incident Zip'].fillna(11000) // 100).astype(int)
        data['Zip_District'] = (data['Incident Zip'].fillna(11000) // 10).astype(int)
        
        # Características de frecuencia por ubicación
        borough_freq = data['Borough'].value_counts()
        data['Borough_Frequency'] = data['Borough'].map(borough_freq)
        
        location_freq = data['Location Type'].value_counts()
        data['Location_Frequency'] = data['Location Type'].map(location_freq).fillna(0)
        
        # Características de agencia
        data['Is_NYPD'] = (data['Agency'] == 'NYPD').astype(int)
        
        # Características de resolución
        data['Closed Date'] = pd.to_datetime(data['Closed Date'], errors='coerce')
        data['Resolution_Hours'] = (data['Closed Date'] - data['Created Date']).dt.total_seconds() / 3600
        data['Has_Resolution'] = (~data['Resolution_Hours'].isna()).astype(int)
        data['Fast_Resolution'] = (data['Resolution_Hours'] <= 24).astype(int)
        data['Very_Fast_Resolution'] = (data['Resolution_Hours'] <= 4).astype(int)
        
        # Características de status
        data['Is_Closed'] = (data['Status'] == 'Closed').astype(int)
        
        print("✓ Características temporales avanzadas")
        print("✓ Características geográficas detalladas")
        print("✓ Características de frecuencia y agencia")
        print("✓ Características de resolución")
        
        # 4. Seleccionar características
        features = [
            # Temporales
            'Hour', 'DayOfWeek', 'Month', 'Quarter', 
            'IsWeekend', 'IsBusinessHour', 'IsEveningHour', 'IsNightHour',
            
            # Geográficas
            'Borough', 'Zip_Region', 'Borough_Frequency', 'Location_Frequency',
            
            # Ubicación
            'Has_Coordinates', 'Has_Address', 'Has_Zip',
            
            # Agencia y resolución
            'Is_NYPD', 'Has_Resolution', 'Fast_Resolution', 'Very_Fast_Resolution', 'Is_Closed'
        ]
        
        # Preparar dataset
        model_data = data[features + ['Priority']].dropna(subset=['Priority'])
        X = model_data[features]
        y = model_data['Priority']
        
        print(f"\n✓ Dataset preparado: {len(model_data):,} filas, {len(features)} características")
        
        # 5. División de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n✓ División completada:")
        print(f"  - Train: {X_train.shape[0]:,} casos")
        print(f"  - Test: {X_test.shape[0]:,} casos")
        
        print("\nDistribución en train:")
        for priority, count in y_train.value_counts().items():
            pct = count / len(y_train) * 100
            print(f"  - {priority}: {count:,} ({pct:.1f}%)")
        
        # 6. Crear pipeline robusto
        print("\n4. CREANDO PIPELINE ROBUSTO...")
        print("-" * 45)
        
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
        
        # 7. Entrenar modelos binarios optimizados
        print("\n5. ENTRENANDO MODELOS BINARIOS...")
        print("-" * 45)
        
        models = {
            'Random Forest': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(
                    n_estimators=300, max_depth=20, min_samples_split=5,
                    min_samples_leaf=2, class_weight='balanced',
                    random_state=42, n_jobs=-1
                ))
            ]),
            
            'Gradient Boosting': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=200, max_depth=10, learning_rate=0.1,
                    min_samples_split=5, min_samples_leaf=2,
                    random_state=42
                ))
            ]),
            
            'Logistic Regression': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(
                    class_weight='balanced', random_state=42, 
                    max_iter=1000, C=1.0
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
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]  # Probabilidad clase positiva
            
            # Calcular métricas
            accuracy = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            auc_score = roc_auc_score(y_test == 'CRITICO', y_pred_proba)
            
            results[name] = {
                'model': pipeline,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'accuracy': accuracy,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro,
                'auc_score': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"✓ CV F1-Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"✓ Test Accuracy: {accuracy:.4f}")
            print(f"✓ Test F1-weighted: {f1_weighted:.4f}")
            print(f"✓ Test AUC: {auc_score:.4f}")
        
        # 8. Evaluar mejor modelo
        print("\n6. EVALUACIÓN DEL MEJOR MODELO...")
        print("-" * 45)
        
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_weighted'])
        best_model_info = results[best_model_name]
        
        print(f"🏆 MEJOR MODELO: {best_model_name}")
        print(f"   CV F1-Score: {best_model_info['cv_mean']:.4f}")
        print(f"   Test Accuracy: {best_model_info['accuracy']:.4f}")
        print(f"   Test F1-weighted: {best_model_info['f1_weighted']:.4f}")
        print(f"   Test AUC: {best_model_info['auc_score']:.4f}")
        
        # Reporte detallado
        print(f"\n📊 REPORTE DETALLADO:")
        print("-" * 30)
        y_pred_best = best_model_info['predictions']
        print(classification_report(y_test, y_pred_best))
        
        # 9. Visualizaciones avanzadas
        print("\n7. CREANDO VISUALIZACIONES AVANZADAS...")
        print("-" * 45)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Distribución binaria
        colors = ['#ff4444', '#44ff44']
        priority_counts.plot(kind='bar', ax=axes[0, 0], color=colors)
        axes[0, 0].set_title('Distribución Binaria')
        axes[0, 0].set_xlabel('Prioridad')
        axes[0, 0].set_ylabel('Número de Casos')
        axes[0, 0].tick_params(axis='x', rotation=0)
        
        # Añadir valores en las barras
        for i, (priority, count) in enumerate(priority_counts.items()):
            axes[0, 0].text(i, count + count*0.01, f'{count:,}', 
                           ha='center', va='bottom', fontweight='bold')
        
        # 2. Matriz de confusión
        cm = confusion_matrix(y_test, y_pred_best)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 1], cmap='Blues',
                    xticklabels=['CRÍTICO', 'NO CRÍTICO'],
                    yticklabels=['CRÍTICO', 'NO CRÍTICO'])
        axes[0, 1].set_title(f'Matriz de Confusión\n{best_model_name}')
        axes[0, 1].set_xlabel('Predicción')
        axes[0, 1].set_ylabel('Real')
        
        # 3. Curva ROC
        y_test_binary = (y_test == 'CRITICO').astype(int)
        fpr, tpr, _ = roc_curve(y_test_binary, best_model_info['probabilities'])
        
        axes[0, 2].plot(fpr, tpr, linewidth=2, label=f'AUC = {best_model_info["auc_score"]:.3f}')
        axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 2].set_xlabel('Tasa de Falsos Positivos')
        axes[0, 2].set_ylabel('Tasa de Verdaderos Positivos')
        axes[0, 2].set_title('Curva ROC')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Comparación de modelos
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        f1_scores = [results[name]['f1_weighted'] for name in model_names]
        auc_scores = [results[name]['auc_score'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.25
        
        axes[1, 0].bar(x - width, accuracies, width, label='Accuracy', alpha=0.8)
        axes[1, 0].bar(x, f1_scores, width, label='F1-Score', alpha=0.8)
        axes[1, 0].bar(x + width, auc_scores, width, label='AUC', alpha=0.8)
        
        axes[1, 0].set_xlabel('Modelos')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Comparación de Modelos')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(model_names, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Distribución de probabilidades
        probas_critico = best_model_info['probabilities'][y_test == 'CRITICO']
        probas_no_critico = best_model_info['probabilities'][y_test == 'NO_CRITICO']
        
        axes[1, 1].hist(probas_critico, bins=20, alpha=0.7, label='CRÍTICO real', color='red')
        axes[1, 1].hist(probas_no_critico, bins=20, alpha=0.7, label='NO CRÍTICO real', color='green')
        axes[1, 1].set_xlabel('Probabilidad de ser CRÍTICO')
        axes[1, 1].set_ylabel('Frecuencia')
        axes[1, 1].set_title('Distribución de Probabilidades')
        axes[1, 1].legend()
        
        # 6. Real vs Predicho
        real_counts = pd.Series(y_test).value_counts()
        pred_counts = pd.Series(y_pred_best).value_counts()
        
        comparison_df = pd.DataFrame({
            'Real': real_counts,
            'Predicho': pred_counts
        }).fillna(0)
        
        comparison_df.plot(kind='bar', ax=axes[1, 2])
        axes[1, 2].set_title('Real vs Predicho')
        axes[1, 2].set_xlabel('Prioridad')
        axes[1, 2].set_ylabel('Número de Casos')
        axes[1, 2].tick_params(axis='x', rotation=0)
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('binary_severity_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Visualizaciones guardadas como 'binary_severity_results.png'")
        
        # 10. Resumen final
        print("\n" + "=" * 65)
        print("RESUMEN FINAL - CLASIFICACIÓN BINARIA")
        print("=" * 65)
        print(f"🏆 Mejor modelo: {best_model_name}")
        print(f"📊 Test Accuracy: {best_model_info['accuracy']:.4f}")
        print(f"📊 Test F1-weighted: {best_model_info['f1_weighted']:.4f}")
        print(f"📊 Test F1-macro: {best_model_info['f1_macro']:.4f}")
        print(f"📊 Test AUC: {best_model_info['auc_score']:.4f}")
        
        # Comparación con modelos anteriores
        original_f1 = 0.5057
        severity_f1 = 0.4478
        
        improvement_vs_original = (best_model_info['f1_weighted'] - original_f1) / original_f1 * 100
        improvement_vs_severity = (best_model_info['f1_weighted'] - severity_f1) / severity_f1 * 100
        
        print(f"\n📈 COMPARACIONES:")
        print(f"   Modelo original (7 clases): {original_f1:.4f}")
        print(f"   Modelo severidad (3 clases): {severity_f1:.4f}")
        print(f"   Modelo binario (2 clases): {best_model_info['f1_weighted']:.4f}")
        print(f"   Mejora vs original: {improvement_vs_original:+.1f}%")
        print(f"   Mejora vs severidad: {improvement_vs_severity:+.1f}%")
        
        if best_model_info['f1_weighted'] > max(original_f1, severity_f1):
            print("\n✅ ¡MODELO BINARIO ES EL MEJOR!")
        elif best_model_info['f1_weighted'] > severity_f1:
            print("\n✅ ¡MODELO BINARIO SUPERA AL DE SEVERIDAD!")
        
        print(f"\n🎯 VENTAJAS DEL ENFOQUE BINARIO:")
        print(f"   ✅ Máxima simplicidad: CRÍTICO vs NO CRÍTICO")
        print(f"   ✅ Balance mejorado: {priority_counts['CRITICO']/priority_counts.sum()*100:.1f}% vs {priority_counts['NO_CRITICO']/priority_counts.sum()*100:.1f}%")
        print(f"   ✅ Interpretación clara para emergencias")
        print(f"   ✅ AUC alto: {best_model_info['auc_score']:.3f}")
        print(f"   ✅ Fácil implementación en sistemas de alerta")
        
        return best_model_info['model']
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    model = main() 