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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def main():
    """Modelo binario CRITICO vs NO_CRITICO"""
    
    print("=" * 60)
    print("MODELO BINARIO: CRÍTICO vs NO CRÍTICO")
    print("=" * 60)
    
    try:
        # 1. Clasificación binaria
        print("\n1. DEFINIENDO CLASIFICACIÓN BINARIA...")
        print("-" * 40)
        
        binary_mapping = {
            'Tortured': 'CRITICO',
            'Chained': 'CRITICO',
            'Neglected': 'NO_CRITICO',
            'No Shelter': 'NO_CRITICO',
            'In Car': 'NO_CRITICO',
            'Noise, Barking Dog (NR5)': 'NO_CRITICO',
            'Other (complaint details)': 'NO_CRITICO'
        }
        
        print("🔴 CRÍTICO (Requiere intervención inmediata):")
        print("   - Tortured, - Chained")
        print("\n🟢 NO CRÍTICO (Otros casos):")
        print("   - Neglected, No Shelter, In Car, Noise, Other")
        
        # 2. Cargar datos
        print("\n2. CARGANDO DATOS...")
        print("-" * 40)
        
        df = pd.read_csv('Animal_Abuse.csv/Animal_Abuse.csv')
        print(f"✓ Dataset cargado: {df.shape[0]:,} filas")
        
        # Aplicar clasificación
        data = df.copy()
        data['Priority'] = data['Descriptor'].map(binary_mapping)
        
        # Mostrar distribución
        priority_counts = data['Priority'].value_counts()
        for priority, count in priority_counts.items():
            pct = count / len(data) * 100
            print(f"  - {priority}: {count:,} ({pct:.2f}%)")
        
        # 3. Crear características
        print("\n3. CREANDO CARACTERÍSTICAS...")
        print("-" * 40)
        
        data['Created Date'] = pd.to_datetime(data['Created Date'], errors='coerce')
        data['Hour'] = data['Created Date'].dt.hour
        data['DayOfWeek'] = data['Created Date'].dt.dayofweek
        data['Month'] = data['Created Date'].dt.month
        data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)
        data['IsBusinessHour'] = data['Hour'].between(9, 17).astype(int)
        data['IsNightHour'] = data['Hour'].between(22, 6).astype(int)
        
        data['Has_Coordinates'] = (~data['Latitude'].isna()).astype(int)
        data['Has_Address'] = (~data['Incident Address'].isna()).astype(int)
        
        borough_freq = data['Borough'].value_counts()
        data['Borough_Frequency'] = data['Borough'].map(borough_freq)
        
        features = [
            'Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'IsBusinessHour', 'IsNightHour',
            'Borough', 'Borough_Frequency', 'Has_Coordinates', 'Has_Address'
        ]
        
        # Preparar dataset
        model_data = data[features + ['Priority']].dropna(subset=['Priority'])
        X = model_data[features]
        y = model_data['Priority']
        
        print(f"✓ Dataset preparado: {len(model_data):,} filas, {len(features)} características")
        
        # 4. División
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n✓ División completada: Train {X_train.shape[0]:,}, Test {X_test.shape[0]:,}")
        
        # 5. Pipeline
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
        
        # 6. Entrenar modelos
        print("\n4. ENTRENANDO MODELOS...")
        print("-" * 40)
        
        models = {
            'Random Forest': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(
                    n_estimators=200, max_depth=15, class_weight='balanced',
                    random_state=42, n_jobs=-1
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
                    class_weight='balanced', random_state=42, max_iter=1000
                ))
            ])
        }
        
        results = {}
        
        for name, pipeline in models.items():
            print(f"\n🔄 Entrenando {name}...")
            
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1_weighted')
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # Calcular probabilidades para AUC
            try:
                y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                auc_score = roc_auc_score(y_test == 'CRITICO', y_pred_proba)
            except:
                auc_score = 0.5
            
            accuracy = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            
            results[name] = {
                'model': pipeline,
                'cv_mean': cv_scores.mean(),
                'accuracy': accuracy,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro,
                'auc_score': auc_score,
                'predictions': y_pred
            }
            
            print(f"✓ CV F1: {cv_scores.mean():.4f}")
            print(f"✓ Accuracy: {accuracy:.4f}")
            print(f"✓ F1-weighted: {f1_weighted:.4f}")
            print(f"✓ AUC: {auc_score:.4f}")
        
        # 7. Mejor modelo
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_weighted'])
        best_model_info = results[best_model_name]
        
        print(f"\n5. MEJOR MODELO: {best_model_name}")
        print("-" * 40)
        print(f"   Accuracy: {best_model_info['accuracy']:.4f}")
        print(f"   F1-weighted: {best_model_info['f1_weighted']:.4f}")
        print(f"   AUC: {best_model_info['auc_score']:.4f}")
        
        print(f"\n📊 REPORTE DETALLADO:")
        y_pred_best = best_model_info['predictions']
        print(classification_report(y_test, y_pred_best))
        
        # 8. Visualizaciones
        print("\n6. CREANDO VISUALIZACIONES...")
        print("-" * 40)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Distribución
        colors = ['#ff4444', '#44ff44']
        priority_counts.plot(kind='bar', ax=axes[0, 0], color=colors)
        axes[0, 0].set_title('Distribución Binaria')
        axes[0, 0].set_xlabel('Prioridad')
        axes[0, 0].tick_params(axis='x', rotation=0)
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred_best)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 1], cmap='Blues')
        axes[0, 1].set_title(f'Matriz de Confusión\n{best_model_name}')
        
        # Comparación de modelos
        model_names = list(results.keys())
        f1_scores = [results[name]['f1_weighted'] for name in model_names]
        
        axes[1, 0].bar(model_names, f1_scores)
        axes[1, 0].set_title('F1-Score por Modelo')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Real vs Predicho
        real_counts = pd.Series(y_test).value_counts()
        pred_counts = pd.Series(y_pred_best).value_counts()
        comparison_df = pd.DataFrame({
            'Real': real_counts,
            'Predicho': pred_counts
        }).fillna(0)
        
        comparison_df.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Real vs Predicho')
        axes[1, 1].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig('binary_model_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Visualizaciones guardadas como 'binary_model_results.png'")
        
        # 9. Resumen final
        print("\n" + "=" * 60)
        print("RESUMEN FINAL - MODELO BINARIO")
        print("=" * 60)
        print(f"🏆 Mejor modelo: {best_model_name}")
        print(f"📊 Accuracy: {best_model_info['accuracy']:.4f}")
        print(f"📊 F1-weighted: {best_model_info['f1_weighted']:.4f}")
        print(f"📊 AUC: {best_model_info['auc_score']:.4f}")
        
        # Comparaciones
        original_f1 = 0.5057
        severity_f1 = 0.4478
        binary_f1 = best_model_info['f1_weighted']
        
        print(f"\n📈 COMPARACIÓN DE ENFOQUES:")
        print(f"   Original (7 clases): {original_f1:.4f}")
        print(f"   Severidad (3 clases): {severity_f1:.4f}")
        print(f"   Binario (2 clases): {binary_f1:.4f}")
        
        if binary_f1 > max(original_f1, severity_f1):
            print(f"\n✅ ¡MODELO BINARIO ES EL MEJOR!")
            improvement = (binary_f1 - max(original_f1, severity_f1)) / max(original_f1, severity_f1) * 100
            print(f"   Mejora: +{improvement:.1f}%")
        
        print(f"\n🎯 VENTAJAS DEL MODELO BINARIO:")
        print(f"   ✅ Simplicidad máxima: 2 clases")
        print(f"   ✅ Balance: {priority_counts['CRITICO']/(priority_counts['CRITICO']+priority_counts['NO_CRITICO'])*100:.1f}% crítico")
        print(f"   ✅ Aplicación práctica para emergencias")
        print(f"   ✅ Fácil implementación")
        
        return best_model_info['model']
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    model = main() 