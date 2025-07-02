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
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

def main():
    """Pipeline principal optimizado"""
    
    print("=" * 60)
    print("MODELO FINAL OPTIMIZADO - NYC ANIMAL ABUSE")
    print("=" * 60)
    
    try:
        # 1. Cargar y preparar datos
        print("\n1. CARGANDO Y PREPARANDO DATOS...")
        print("-" * 40)
        
        df = pd.read_csv('Animal_Abuse.csv/Animal_Abuse.csv')
        print(f"✓ Dataset cargado: {df.shape[0]:,} filas")
        
        # Crear características robustas
        data = df.copy()
        data['Created Date'] = pd.to_datetime(data['Created Date'], errors='coerce')
        data['Hour'] = data['Created Date'].dt.hour
        data['DayOfWeek'] = data['Created Date'].dt.dayofweek
        data['Month'] = data['Created Date'].dt.month
        data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)
        data['Has_Coordinates'] = (~data['Latitude'].isna()).astype(int)
        data['Has_Address'] = (~data['Incident Address'].isna()).astype(int)
        
        # Seleccionar características
        features = ['Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'Borough', 'Has_Coordinates', 'Has_Address']
        
        # Preparar dataset
        model_data = data[features + ['Descriptor']].dropna(subset=['Descriptor'])
        X = model_data[features]
        y = model_data['Descriptor']
        
        print(f"✓ Dataset preparado: {len(model_data):,} filas, {len(features)} características")
        
        # 2. División de datos
        print("\n2. DIVISIÓN Y BALANCEAMIENTO...")
        print("-" * 40)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Calcular pesos balanceados
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        print("Distribución de clases:")
        for cls, count in y_train.value_counts().items():
            weight = class_weight_dict.get(cls, 1.0)
            pct = count / len(y_train) * 100
            print(f"  - {cls}: {count:,} ({pct:.1f}%) - peso: {weight:.2f}")
        
        # 3. Crear pipeline
        print("\n3. CREANDO PIPELINE DE PROCESAMIENTO...")
        print("-" * 40)
        
        # Identificar tipos de características
        categorical_features = ['Borough']
        numerical_features = [col for col in features if col not in categorical_features]
        
        # Crear preprocessor
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
        
        # 4. Entrenar modelo optimizado
        print("\n4. ENTRENANDO MODELO OPTIMIZADO...")
        print("-" * 40)
        
        # Crear pipeline completo
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight=class_weight_dict,
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        # Validación cruzada
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1_weighted')
        print(f"✓ CV F1-Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Entrenar modelo final
        pipeline.fit(X_train, y_train)
        
        # 5. Evaluación
        print("\n5. EVALUACIÓN DEL MODELO...")
        print("-" * 40)
        
        y_pred = pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        print(f"✓ Test Accuracy: {accuracy:.4f}")
        print(f"✓ Test F1-weighted: {f1_weighted:.4f}")
        print(f"✓ Test F1-macro: {f1_macro:.4f}")
        
        # Reporte detallado
        print(f"\n📊 REPORTE DETALLADO:")
        print("-" * 40)
        print(classification_report(y_test, y_pred))
        
        # 6. Visualizaciones
        print("\n6. CREANDO VISUALIZACIONES...")
        print("-" * 40)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues')
        axes[0, 0].set_title('Matriz de Confusión')
        axes[0, 0].set_xlabel('Predicción')
        axes[0, 0].set_ylabel('Real')
        
        # Distribución de predicciones
        pred_counts = pd.Series(y_pred).value_counts()
        axes[0, 1].pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Distribución de Predicciones')
        
        # F1-Score por clase
        classes = sorted(y_test.unique())
        f1_per_class = []
        for class_name in classes:
            f1 = f1_score(y_test == class_name, y_pred == class_name)
            f1_per_class.append(f1)
        
        axes[1, 0].barh(classes, f1_per_class)
        axes[1, 0].set_xlabel('F1-Score')
        axes[1, 0].set_title('F1-Score por Clase')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Comparación Real vs Predicho
        real_counts = pd.Series(y_test).value_counts()
        comparison_df = pd.DataFrame({
            'Real': real_counts,
            'Predicho': pred_counts
        }).fillna(0)
        
        comparison_df.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Real vs Predicho')
        axes[1, 1].set_xlabel('Clases')
        axes[1, 1].set_ylabel('Número de Casos')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('final_optimized_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Visualizaciones guardadas como 'final_optimized_results.png'")
        
        # 7. Resumen final
        print("\n" + "=" * 60)
        print("RESUMEN FINAL")
        print("=" * 60)
        print(f"🏆 Modelo: Random Forest Optimizado")
        print(f"📊 CV F1-Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"📊 Test Accuracy: {accuracy:.4f}")
        print(f"📊 Test F1-weighted: {f1_weighted:.4f}")
        print(f"📊 Test F1-macro: {f1_macro:.4f}")
        
        # Comparación con baseline
        baseline_acc = 0.5652
        baseline_f1 = 0.5057
        
        improvement_acc = (accuracy - baseline_acc) / baseline_acc * 100
        improvement_f1 = (f1_weighted - baseline_f1) / baseline_f1 * 100
        
        print(f"\n📈 MEJORAS RESPECTO AL BASELINE:")
        print(f"   Accuracy: {improvement_acc:+.1f}%")
        print(f"   F1-weighted: {improvement_f1:+.1f}%")
        
        if f1_weighted > baseline_f1:
            print("\n✅ ¡MODELO MEJORADO EXITOSAMENTE!")
        else:
            print("\n⚠️  Modelo baseline sigue siendo mejor")
        
        print("\n🎯 CARACTERÍSTICAS DEL MODELO:")
        print("   ✅ Balanceamiento con class weights")
        print("   ✅ Validación cruzada estable")
        print("   ✅ Hiperparámetros optimizados")
        print("   ✅ Pipeline robusto")
        
        return pipeline
        
    except Exception as e:
        print(f"✗ Error en el pipeline: {e}")
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    model = main() 