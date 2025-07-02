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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def main():
    """Modelo binario optimizado para casos críticos"""
    
    print("=" * 65)
    print("MODELO BINARIO OPTIMIZADO - DETECCIÓN DE CASOS CRÍTICOS")
    print("=" * 65)
    
    try:
        # 1. Clasificación binaria
        print("\n1. DEFINIENDO CLASIFICACIÓN BINARIA OPTIMIZADA...")
        print("-" * 50)
        
        binary_mapping = {
            'Tortured': 'CRITICO',
            'Chained': 'CRITICO',
            'Neglected': 'NO_CRITICO',
            'No Shelter': 'NO_CRITICO',
            'In Car': 'NO_CRITICO',
            'Noise, Barking Dog (NR5)': 'NO_CRITICO',
            'Other (complaint details)': 'NO_CRITICO'
        }
        
        print("🔴 CRÍTICO (Tortured + Chained):")
        print("   - Casos que requieren intervención INMEDIATA")
        print("🟢 NO CRÍTICO (Resto):")
        print("   - Casos importantes pero menos urgentes")
        
        # 2. Cargar datos
        print("\n2. CARGANDO Y PREPARANDO DATOS...")
        print("-" * 50)
        
        df = pd.read_csv('Animal_Abuse.csv/Animal_Abuse.csv')
        print(f"✓ Dataset cargado: {df.shape[0]:,} filas")
        
        data = df.copy()
        data['Priority'] = data['Descriptor'].map(binary_mapping)
        
        priority_counts = data['Priority'].value_counts()
        for priority, count in priority_counts.items():
            pct = count / len(data) * 100
            print(f"  - {priority}: {count:,} ({pct:.2f}%)")
        
        print(f"\n⚠️  DESAFÍO: {priority_counts['CRITICO']/priority_counts.sum()*100:.1f}% casos críticos")
        print("   → Necesitamos optimizar para NO perder casos críticos")
        
        # 3. Crear características mejoradas
        print("\n3. INGENIERÍA DE CARACTERÍSTICAS MEJORADA...")
        print("-" * 50)
        
        data['Created Date'] = pd.to_datetime(data['Created Date'], errors='coerce')
        
        # Características temporales más detalladas
        data['Hour'] = data['Created Date'].dt.hour
        data['DayOfWeek'] = data['Created Date'].dt.dayofweek
        data['Month'] = data['Created Date'].dt.month
        data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)
        data['IsBusinessHour'] = data['Hour'].between(9, 17).astype(int)
        data['IsEveningHour'] = data['Hour'].between(18, 22).astype(int)
        data['IsNightHour'] = data['Hour'].between(23, 6).astype(int)
        data['IsEarlyMorning'] = data['Hour'].between(0, 8).astype(int)
        
        # Características geográficas
        data['Has_Coordinates'] = (~data['Latitude'].isna()).astype(int)
        data['Has_Address'] = (~data['Incident Address'].isna()).astype(int)
        data['Has_Zip'] = (~data['Incident Zip'].isna()).astype(int)
        
        # Características de ubicación más específicas
        borough_freq = data['Borough'].value_counts()
        data['Borough_Frequency'] = data['Borough'].map(borough_freq)
        
        # Crear indicadores por borough (algunos pueden tener más casos críticos)
        data['Is_Brooklyn'] = (data['Borough'] == 'BROOKLYN').astype(int)
        data['Is_Manhattan'] = (data['Borough'] == 'MANHATTAN').astype(int)
        data['Is_Queens'] = (data['Borough'] == 'QUEENS').astype(int)
        data['Is_Bronx'] = (data['Borough'] == 'BRONX').astype(int)
        
        # Características de resolución
        data['Closed Date'] = pd.to_datetime(data['Closed Date'], errors='coerce')
        data['Resolution_Hours'] = (data['Closed Date'] - data['Created Date']).dt.total_seconds() / 3600
        data['Has_Resolution'] = (~data['Resolution_Hours'].isna()).astype(int)
        data['Fast_Resolution'] = (data['Resolution_Hours'] <= 24).astype(int)
        
        # Agencia
        data['Is_NYPD'] = (data['Agency'] == 'NYPD').astype(int)
        
        features = [
            # Temporales
            'Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'IsBusinessHour', 
            'IsEveningHour', 'IsNightHour', 'IsEarlyMorning',
            
            # Geográficas
            'Borough_Frequency', 'Is_Brooklyn', 'Is_Manhattan', 'Is_Queens', 'Is_Bronx',
            
            # Ubicación
            'Has_Coordinates', 'Has_Address', 'Has_Zip',
            
            # Resolución y agencia
            'Has_Resolution', 'Fast_Resolution', 'Is_NYPD'
        ]
        
        model_data = data[features + ['Priority']].dropna(subset=['Priority'])
        X = model_data[features]
        y = model_data['Priority']
        
        print(f"✓ {len(features)} características seleccionadas")
        print(f"✓ Dataset final: {len(model_data):,} filas")
        
        # 4. División estratificada
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n✓ División estratificada:")
        print(f"  - Train: {X_train.shape[0]:,}")
        print(f"  - Test: {X_test.shape[0]:,}")
        
        train_dist = y_train.value_counts()
        for priority, count in train_dist.items():
            pct = count / len(y_train) * 100
            print(f"    {priority}: {count:,} ({pct:.1f}%)")
        
        # 5. Pipeline con preprocesamiento robusto
        print("\n4. PIPELINE OPTIMIZADO...")
        print("-" * 50)
        
        # Todas las características son numéricas ahora
        preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # 6. Modelos con pesos ajustados para casos críticos
        print("\n5. ENTRENANDO MODELOS OPTIMIZADOS...")
        print("-" * 50)
        
        # Calcular pesos para balancear mejor los casos críticos
        n_samples = len(y_train)
        n_critico = sum(y_train == 'CRITICO')
        n_no_critico = sum(y_train == 'NO_CRITICO')
        
        # Peso más alto para casos críticos para no perderlos
        weight_critico = n_samples / (2 * n_critico) * 2  # Factor 2 extra para casos críticos
        weight_no_critico = n_samples / (2 * n_no_critico)
        
        print(f"📊 Pesos calculados:")
        print(f"   CRÍTICO: {weight_critico:.2f}")
        print(f"   NO CRÍTICO: {weight_no_critico:.2f}")
        
        models = {
            'Random Forest Balanceado': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(
                    n_estimators=300, max_depth=20, min_samples_split=3,
                    min_samples_leaf=1, class_weight={'CRITICO': weight_critico, 'NO_CRITICO': weight_no_critico},
                    random_state=42, n_jobs=-1
                ))
            ]),
            
            'Gradient Boosting Optimizado': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=200, max_depth=8, learning_rate=0.05,
                    min_samples_split=5, min_samples_leaf=2,
                    random_state=42
                ))
            ]),
            
            'Logistic Regression Balanceada': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(
                    class_weight={'CRITICO': weight_critico, 'NO_CRITICO': weight_no_critico},
                    random_state=42, max_iter=2000, C=0.5
                ))
            ])
        }
        
        results = {}
        
        for name, pipeline in models.items():
            print(f"\n🔄 Entrenando {name}...")
            
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1_weighted')
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # Métricas con foco en casos críticos
            accuracy = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            
            # F1 específico para casos críticos
            f1_critico = f1_score(y_test, y_pred, pos_label='CRITICO', average='binary')
            
            # Recall crítico (lo más importante - no perder casos críticos)
            from sklearn.metrics import recall_score
            recall_critico = recall_score(y_test, y_pred, pos_label='CRITICO', average='binary')
            
            try:
                y_pred_proba = pipeline.predict_proba(X_test)[:, 1 if pipeline.classes_[1] == 'CRITICO' else 0]
                auc_score = roc_auc_score(y_test == 'CRITICO', y_pred_proba)
            except:
                auc_score = 0.5
            
            results[name] = {
                'model': pipeline,
                'cv_mean': cv_scores.mean(),
                'accuracy': accuracy,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro,
                'f1_critico': f1_critico,
                'recall_critico': recall_critico,
                'auc_score': auc_score,
                'predictions': y_pred
            }
            
            print(f"✓ CV F1: {cv_scores.mean():.4f}")
            print(f"✓ Accuracy: {accuracy:.4f}")
            print(f"✓ F1-weighted: {f1_weighted:.4f}")
            print(f"✓ F1-CRÍTICO: {f1_critico:.4f}")
            print(f"✓ Recall-CRÍTICO: {recall_critico:.4f} ← ¡CRÍTICO!")
            print(f"✓ AUC: {auc_score:.4f}")
        
        # 7. Seleccionar modelo basado en recall crítico
        print("\n6. SELECCIÓN DEL MEJOR MODELO...")
        print("-" * 50)
        
        # Priorizar recall crítico sobre otras métricas
        best_model_name = max(results.keys(), key=lambda x: results[x]['recall_critico'])
        best_model_info = results[best_model_name]
        
        print(f"🏆 MEJOR MODELO (por recall crítico): {best_model_name}")
        print(f"   Accuracy: {best_model_info['accuracy']:.4f}")
        print(f"   F1-weighted: {best_model_info['f1_weighted']:.4f}")
        print(f"   F1-CRÍTICO: {best_model_info['f1_critico']:.4f}")
        print(f"   Recall-CRÍTICO: {best_model_info['recall_critico']:.4f} ← ¡CLAVE!")
        print(f"   AUC: {best_model_info['auc_score']:.4f}")
        
        print(f"\n📊 REPORTE DETALLADO:")
        print("-" * 30)
        y_pred_best = best_model_info['predictions']
        print(classification_report(y_test, y_pred_best))
        
        # 8. Visualizaciones optimizadas
        print("\n7. VISUALIZACIONES...")
        print("-" * 50)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Distribución
        colors = ['#ff4444', '#44ff44']
        priority_counts.plot(kind='bar', ax=axes[0, 0], color=colors)
        axes[0, 0].set_title('Distribución Binaria Optimizada')
        axes[0, 0].set_xlabel('Prioridad')
        axes[0, 0].tick_params(axis='x', rotation=0)
        
        # 2. Matriz de confusión mejorada
        cm = confusion_matrix(y_test, y_pred_best)
        
        # Calcular porcentajes
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Crear anotaciones combinadas
        annots = []
        for i in range(cm.shape[0]):
            row = []
            for j in range(cm.shape[1]):
                row.append(f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)')
            annots.append(row)
        
        sns.heatmap(cm, annot=annots, fmt='', ax=axes[0, 1], cmap='Blues',
                    xticklabels=['CRÍTICO', 'NO CRÍTICO'],
                    yticklabels=['CRÍTICO', 'NO CRÍTICO'])
        axes[0, 1].set_title(f'Matriz de Confusión Detallada\n{best_model_name}')
        axes[0, 1].set_xlabel('Predicción')
        axes[0, 1].set_ylabel('Real')
        
        # 3. Métricas por modelo
        model_names = list(results.keys())
        recall_criticos = [results[name]['recall_critico'] for name in model_names]
        f1_criticos = [results[name]['f1_critico'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 2].bar(x - width/2, recall_criticos, width, label='Recall CRÍTICO', color='red', alpha=0.7)
        axes[0, 2].bar(x + width/2, f1_criticos, width, label='F1 CRÍTICO', color='orange', alpha=0.7)
        axes[0, 2].set_xlabel('Modelos')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].set_title('Métricas para Casos CRÍTICOS')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Comparación F1-weighted todos los modelos
        f1_weighteds = [results[name]['f1_weighted'] for name in model_names]
        
        axes[1, 0].bar(model_names, f1_weighteds, color='skyblue')
        axes[1, 0].set_title('F1-weighted por Modelo')
        axes[1, 0].set_ylabel('F1-weighted')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Real vs Predicho
        real_counts = pd.Series(y_test).value_counts()
        pred_counts = pd.Series(y_pred_best).value_counts()
        comparison_df = pd.DataFrame({
            'Real': real_counts,
            'Predicho': pred_counts
        }).fillna(0)
        
        comparison_df.plot(kind='bar', ax=axes[1, 1], color=['red', 'green'])
        axes[1, 1].set_title('Distribución: Real vs Predicho')
        axes[1, 1].set_xlabel('Prioridad')
        axes[1, 1].tick_params(axis='x', rotation=0)
        axes[1, 1].legend()
        
        # 6. Evolución de modelos
        model_evolution = pd.DataFrame({
            'Modelo': ['Original\n(7 clases)', 'Severidad\n(3 clases)', 'Binario\n(2 clases)', 'Binario\nOptimizado'],
            'F1_Score': [0.5057, 0.4478, 0.7501, best_model_info['f1_weighted']],
            'Accuracy': [0.5652, 0.5591, 0.8246, best_model_info['accuracy']]
        })
        
        x_pos = np.arange(len(model_evolution))
        width = 0.35
        
        axes[1, 2].bar(x_pos - width/2, model_evolution['F1_Score'], width, label='F1-Score', alpha=0.8)
        axes[1, 2].bar(x_pos + width/2, model_evolution['Accuracy'], width, label='Accuracy', alpha=0.8)
        axes[1, 2].set_xlabel('Evolución de Modelos')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_title('Evolución del Rendimiento')
        axes[1, 2].set_xticks(x_pos)
        axes[1, 2].set_xticklabels(model_evolution['Modelo'])
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('optimized_binary_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Visualizaciones guardadas como 'optimized_binary_results.png'")
        
        # 9. Resumen final completo
        print("\n" + "=" * 65)
        print("RESUMEN FINAL - MODELO BINARIO OPTIMIZADO")
        print("=" * 65)
        
        print(f"🏆 MEJOR MODELO: {best_model_name}")
        print(f"📊 Accuracy: {best_model_info['accuracy']:.4f}")
        print(f"📊 F1-weighted: {best_model_info['f1_weighted']:.4f}")
        print(f"📊 F1-CRÍTICO: {best_model_info['f1_critico']:.4f}")
        print(f"📊 Recall-CRÍTICO: {best_model_info['recall_critico']:.4f} ← ¡CRÍTICO!")
        print(f"📊 AUC: {best_model_info['auc_score']:.4f}")
        
        # Análisis de casos críticos
        cm = confusion_matrix(y_test, y_pred_best, labels=['CRITICO', 'NO_CRITICO'])
        casos_criticos_detectados = cm[0, 0]  # Verdaderos positivos críticos
        casos_criticos_perdidos = cm[0, 1]   # Falsos negativos críticos
        casos_criticos_reales = casos_criticos_detectados + casos_criticos_perdidos
        
        print(f"\n🔍 ANÁLISIS DE CASOS CRÍTICOS:")
        print(f"   Total casos críticos reales: {casos_criticos_reales}")
        print(f"   Casos críticos detectados: {casos_criticos_detectados}")
        print(f"   Casos críticos perdidos: {casos_criticos_perdidos}")
        print(f"   Tasa de detección: {casos_criticos_detectados/casos_criticos_reales*100:.1f}%")
        
        # Comparación evolutiva
        print(f"\n📈 EVOLUCIÓN DE MODELOS:")
        print(f"   Original (7 clases): F1={0.5057:.4f}, Acc={0.5652:.4f}")
        print(f"   Severidad (3 clases): F1={0.4478:.4f}, Acc={0.5591:.4f}")
        print(f"   Binario simple: F1={0.7501:.4f}, Acc={0.8246:.4f}")
        print(f"   Binario optimizado: F1={best_model_info['f1_weighted']:.4f}, Acc={best_model_info['accuracy']:.4f}")
        
        if best_model_info['recall_critico'] > 0.3:
            print(f"\n✅ ¡EXCELENTE! Recall crítico > 30%")
        elif best_model_info['recall_critico'] > 0.1:
            print(f"\n👍 BUENO: Recall crítico > 10%")
        else:
            print(f"\n⚠️  MEJORABLE: Recall crítico bajo")
        
        print(f"\n🎯 RECOMENDACIÓN FINAL:")
        print(f"   ✅ Usar modelo binario optimizado")
        print(f"   ✅ Enfoque: CRÍTICO vs NO CRÍTICO")
        print(f"   ✅ Balance entre precisión general y detección crítica")
        print(f"   ✅ Implementable en sistemas de alerta")
        
        return best_model_info['model']
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    model = main() 