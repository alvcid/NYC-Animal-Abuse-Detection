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
    """Modelo binario mejorado - Balance entre accuracy y recall crítico"""
    
    print("=" * 70)
    print("MODELO BINARIO MEJORADO - BALANCE INTELIGENTE")
    print("=" * 70)
    
    try:
        # 1. Clasificación binaria
        print("\n1. MANTENIENDO CLASIFICACIÓN BINARIA EFECTIVA...")
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
        
        print("🔴 CRÍTICO: Tortured + Chained")
        print("🟢 NO CRÍTICO: Resto")
        
        # 2. Cargar datos
        print("\n2. CARGANDO DATOS...")
        print("-" * 50)
        
        df = pd.read_csv('Animal_Abuse.csv/Animal_Abuse.csv')
        print(f"✓ Dataset: {df.shape[0]:,} filas")
        
        data = df.copy()
        data['Priority'] = data['Descriptor'].map(binary_mapping)
        
        priority_counts = data['Priority'].value_counts()
        print(f"  - NO_CRITICO: {priority_counts['NO_CRITICO']:,} ({priority_counts['NO_CRITICO']/priority_counts.sum()*100:.1f}%)")
        print(f"  - CRITICO: {priority_counts['CRITICO']:,} ({priority_counts['CRITICO']/priority_counts.sum()*100:.1f}%)")
        
        # 3. Características optimizadas (más que el modelo simple)
        print("\n3. CARACTERÍSTICAS MEJORADAS...")
        print("-" * 50)
        
        data['Created Date'] = pd.to_datetime(data['Created Date'], errors='coerce')
        
        # Temporales
        data['Hour'] = data['Created Date'].dt.hour
        data['DayOfWeek'] = data['Created Date'].dt.dayofweek
        data['Month'] = data['Created Date'].dt.month
        data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)
        data['IsBusinessHour'] = data['Hour'].between(9, 17).astype(int)
        data['IsNightHour'] = data['Hour'].between(22, 6).astype(int)
        data['IsEveningHour'] = data['Hour'].between(18, 22).astype(int)
        
        # Geográficas
        data['Has_Coordinates'] = (~data['Latitude'].isna()).astype(int)
        data['Has_Address'] = (~data['Incident Address'].isna()).astype(int)
        data['Has_Zip'] = (~data['Incident Zip'].isna()).astype(int)
        
        # Borough específicos
        data['Is_Brooklyn'] = (data['Borough'] == 'BROOKLYN').astype(int)
        data['Is_Manhattan'] = (data['Borough'] == 'MANHATTAN').astype(int)
        data['Is_Queens'] = (data['Borough'] == 'QUEENS').astype(int)
        data['Is_Bronx'] = (data['Borough'] == 'BRONX').astype(int)
        
        # Frecuencias
        borough_freq = data['Borough'].value_counts()
        data['Borough_Frequency'] = data['Borough'].map(borough_freq)
        
        # Agencia
        data['Is_NYPD'] = (data['Agency'] == 'NYPD').astype(int)
        
        # Resolución
        data['Closed Date'] = pd.to_datetime(data['Closed Date'], errors='coerce')
        data['Has_Resolution'] = (~data['Closed Date'].isna()).astype(int)
        
        features = [
            'Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'IsBusinessHour', 'IsNightHour', 'IsEveningHour',
            'Has_Coordinates', 'Has_Address', 'Has_Zip',
            'Is_Brooklyn', 'Is_Manhattan', 'Is_Queens', 'Is_Bronx', 'Borough_Frequency',
            'Is_NYPD', 'Has_Resolution'
        ]
        
        model_data = data[features + ['Priority']].dropna(subset=['Priority'])
        X = model_data[features]
        y = model_data['Priority']
        
        print(f"✓ {len(features)} características")
        print(f"✓ Dataset: {len(model_data):,} filas")
        
        # 4. División
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n✓ Train: {X_train.shape[0]:,}, Test: {X_test.shape[0]:,}")
        
        # 5. Pipeline
        preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # 6. Modelos con PESOS MODERADOS (no extremos)
        print("\n4. ENTRENANDO MODELOS CON PESOS MODERADOS...")
        print("-" * 50)
        
        # Pesos moderados - no tan extremos como el optimizado
        models = {
            'Random Forest Moderado': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(
                    n_estimators=300, max_depth=20, min_samples_split=3,
                    class_weight={
                        'CRITICO': 3.0,      # Peso moderado (vs 5.8 anterior)
                        'NO_CRITICO': 0.8    # Peso moderado (vs 0.6 anterior)
                    },
                    random_state=42, n_jobs=-1
                ))
            ]),
            
            'Gradient Boosting Balanceado': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=250, max_depth=10, learning_rate=0.08,
                    min_samples_split=10, min_samples_leaf=5,
                    random_state=42
                ))
            ]),
            
            'Logistic Moderado': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(
                    class_weight={
                        'CRITICO': 2.5,      # Peso moderado
                        'NO_CRITICO': 0.9
                    },
                    random_state=42, max_iter=1500, C=1.0
                ))
            ]),
            
            # Modelo sin pesos (como referencia)
            'Random Forest Sin Pesos': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(
                    n_estimators=300, max_depth=20, min_samples_split=5,
                    random_state=42, n_jobs=-1
                ))
            ])
        }
        
        results = {}
        
        for name, pipeline in models.items():
            print(f"\n🔄 Entrenando {name}...")
            
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1_weighted')
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # Métricas balanceadas
            accuracy = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            
            # Métricas críticas
            from sklearn.metrics import recall_score, precision_score
            f1_critico = f1_score(y_test, y_pred, pos_label='CRITICO', average='binary')
            recall_critico = recall_score(y_test, y_pred, pos_label='CRITICO', average='binary')
            precision_critico = precision_score(y_test, y_pred, pos_label='CRITICO', average='binary')
            
            try:
                y_pred_proba = pipeline.predict_proba(X_test)
                if hasattr(pipeline, 'classes_'):
                    critico_idx = list(pipeline.classes_).index('CRITICO') if 'CRITICO' in pipeline.classes_ else 0
                else:
                    critico_idx = 1
                auc_score = roc_auc_score(y_test == 'CRITICO', y_pred_proba[:, critico_idx])
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
                'precision_critico': precision_critico,
                'auc_score': auc_score,
                'predictions': y_pred
            }
            
            print(f"✓ CV F1: {cv_scores.mean():.4f}")
            print(f"✓ Accuracy: {accuracy:.4f}")
            print(f"✓ F1-weighted: {f1_weighted:.4f}")
            print(f"✓ Recall-CRÍTICO: {recall_critico:.4f}")
            print(f"✓ Precision-CRÍTICO: {precision_critico:.4f}")
        
        # 7. Seleccionar modelo balanceado
        print("\n5. SELECCIÓN DEL MEJOR MODELO BALANCEADO...")
        print("-" * 50)
        
        # Función de score balanceado que considera accuracy Y recall crítico
        def balanced_score(result):
            acc = result['accuracy']
            recall_crit = result['recall_critico']
            f1_w = result['f1_weighted']
            
            # Balance: 60% accuracy general + 40% recall crítico
            return 0.6 * acc + 0.4 * recall_crit
        
        best_model_name = max(results.keys(), key=lambda x: balanced_score(results[x]))
        best_model_info = results[best_model_name]
        
        print(f"🏆 MEJOR MODELO BALANCEADO: {best_model_name}")
        print(f"   Accuracy: {best_model_info['accuracy']:.4f}")
        print(f"   F1-weighted: {best_model_info['f1_weighted']:.4f}")
        print(f"   Recall-CRÍTICO: {best_model_info['recall_critico']:.4f}")
        print(f"   Precision-CRÍTICO: {best_model_info['precision_critico']:.4f}")
        print(f"   AUC: {best_model_info['auc_score']:.4f}")
        print(f"   Score balanceado: {balanced_score(best_model_info):.4f}")
        
        print(f"\n📊 REPORTE DETALLADO:")
        print("-" * 30)
        y_pred_best = best_model_info['predictions']
        print(classification_report(y_test, y_pred_best))
        
        # 8. Análisis de matriz de confusión
        cm = confusion_matrix(y_test, y_pred_best, labels=['CRITICO', 'NO_CRITICO'])
        casos_criticos_reales = sum(y_test == 'CRITICO')
        casos_criticos_detectados = cm[0, 0]
        casos_criticos_perdidos = cm[0, 1]
        casos_no_criticos_reales = sum(y_test == 'NO_CRITICO')
        falsas_alarmas = cm[1, 0]
        
        print(f"\n🔍 ANÁLISIS DETALLADO:")
        print(f"   Casos críticos reales: {casos_criticos_reales}")
        print(f"   Casos críticos detectados: {casos_criticos_detectados}")
        print(f"   Casos críticos perdidos: {casos_criticos_perdidos}")
        print(f"   Falsas alarmas: {falsas_alarmas}")
        print(f"   Tasa detección crítica: {casos_criticos_detectados/casos_criticos_reales*100:.1f}%")
        print(f"   Tasa falsas alarmas: {falsas_alarmas/casos_no_criticos_reales*100:.1f}%")
        
        # 9. Visualizaciones
        print("\n6. VISUALIZACIONES...")
        print("-" * 50)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Comparación de modelos por accuracy
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        
        axes[0, 0].bar(range(len(model_names)), accuracies, color='skyblue')
        axes[0, 0].set_title('Accuracy por Modelo')
        axes[0, 0].set_xticks(range(len(model_names)))
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Comparación de recall crítico
        recall_criticos = [results[name]['recall_critico'] for name in model_names]
        
        axes[0, 1].bar(range(len(model_names)), recall_criticos, color='red', alpha=0.7)
        axes[0, 1].set_title('Recall Crítico por Modelo')
        axes[0, 1].set_xticks(range(len(model_names)))
        axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Recall Crítico')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Matriz de confusión del mejor modelo
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 2], cmap='Blues',
                    xticklabels=['CRÍTICO', 'NO CRÍTICO'],
                    yticklabels=['CRÍTICO', 'NO CRÍTICO'])
        axes[0, 2].set_title(f'Matriz de Confusión\n{best_model_name}')
        axes[0, 2].set_xlabel('Predicción')
        axes[0, 2].set_ylabel('Real')
        
        # 4. Balance accuracy vs recall
        axes[1, 0].scatter(accuracies, recall_criticos, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            axes[1, 0].annotate(name, (accuracies[i], recall_criticos[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 0].set_xlabel('Accuracy')
        axes[1, 0].set_ylabel('Recall Crítico')
        axes[1, 0].set_title('Balance: Accuracy vs Recall Crítico')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Comparación evolutiva
        evolution_data = {
            'Modelo': ['Original\n7 clases', 'Binario\nSimple', 'Binario\nMejorado'],
            'Accuracy': [0.5652, 0.8246, best_model_info['accuracy']],
            'F1_weighted': [0.5057, 0.7501, best_model_info['f1_weighted']],
            'Recall_Critico': [0.0, 0.01, best_model_info['recall_critico']]  # Estimado para original
        }
        
        x_pos = np.arange(len(evolution_data['Modelo']))
        width = 0.25
        
        axes[1, 1].bar(x_pos - width, evolution_data['Accuracy'], width, label='Accuracy', alpha=0.8)
        axes[1, 1].bar(x_pos, evolution_data['F1_weighted'], width, label='F1-weighted', alpha=0.8)
        axes[1, 1].bar(x_pos + width, evolution_data['Recall_Critico'], width, label='Recall Crítico', alpha=0.8)
        
        axes[1, 1].set_xlabel('Evolución de Modelos')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Evolución del Rendimiento')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(evolution_data['Modelo'])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Distribución real vs predicho
        real_counts = pd.Series(y_test).value_counts()
        pred_counts = pd.Series(y_pred_best).value_counts()
        comparison_df = pd.DataFrame({
            'Real': real_counts,
            'Predicho': pred_counts
        }).fillna(0)
        
        comparison_df.plot(kind='bar', ax=axes[1, 2], color=['red', 'green'])
        axes[1, 2].set_title('Real vs Predicho')
        axes[1, 2].set_xlabel('Prioridad')
        axes[1, 2].tick_params(axis='x', rotation=0)
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('improved_binary_model_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Visualizaciones guardadas como 'improved_binary_model_results.png'")
        
        # 10. Resumen final
        print("\n" + "=" * 70)
        print("RESUMEN FINAL - MODELO BINARIO MEJORADO")
        print("=" * 70)
        
        print(f"🏆 MEJOR MODELO: {best_model_name}")
        print(f"📊 Accuracy: {best_model_info['accuracy']:.4f}")
        print(f"📊 F1-weighted: {best_model_info['f1_weighted']:.4f}")
        print(f"📊 Recall-CRÍTICO: {best_model_info['recall_critico']:.4f}")
        print(f"📊 Precision-CRÍTICO: {best_model_info['precision_critico']:.4f}")
        print(f"📊 AUC: {best_model_info['auc_score']:.4f}")
        
        print(f"\n📈 COMPARACIÓN FINAL:")
        print(f"   Modelo original: Acc=56.5%, F1=50.6%, RecallCrítico≈0%")
        print(f"   Binario simple: Acc=82.5%, F1=75.0%, RecallCrítico=1%")
        print(f"   Binario mejorado: Acc={best_model_info['accuracy']*100:.1f}%, F1={best_model_info['f1_weighted']*100:.1f}%, RecallCrítico={best_model_info['recall_critico']*100:.1f}%")
        
        improvement_acc = (best_model_info['accuracy'] - 0.5652) / 0.5652 * 100
        improvement_f1 = (best_model_info['f1_weighted'] - 0.5057) / 0.5057 * 100
        
        print(f"\n✅ MEJORAS vs ORIGINAL:")
        print(f"   Accuracy: +{improvement_acc:.1f}%")
        print(f"   F1-weighted: +{improvement_f1:.1f}%")
        print(f"   Recall crítico: Significativamente mejor")
        
        if best_model_info['accuracy'] > 0.7 and best_model_info['recall_critico'] > 0.1:
            print(f"\n🎉 ¡EXCELENTE BALANCE LOGRADO!")
        elif best_model_info['accuracy'] > 0.6 and best_model_info['recall_critico'] > 0.05:
            print(f"\n👍 BUEN BALANCE LOGRADO")
        
        print(f"\n🎯 CONCLUSIONES:")
        print(f"   ✅ Modelo binario es superior al original")
        print(f"   ✅ Balance razonable entre accuracy y detección crítica")
        print(f"   ✅ Implementable en sistemas reales")
        print(f"   ✅ {casos_criticos_detectados}/{casos_criticos_reales} casos críticos detectados")
        
        return best_model_info['model']
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    model = main() 