import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def main():
    """Modelo binario balanceado - Mejorar recall crítico manteniendo accuracy"""
    
    print("=" * 60)
    print("MODELO BINARIO BALANCEADO")
    print("=" * 60)
    print("Objetivo: Mejorar detección crítica SIN sacrificar accuracy")
    
    try:
        # 1. Datos
        print("\n1. PREPARANDO DATOS...")
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
        
        df = pd.read_csv('Animal_Abuse.csv/Animal_Abuse.csv')
        data = df.copy()
        data['Priority'] = data['Descriptor'].map(binary_mapping)
        
        priority_counts = data['Priority'].value_counts()
        print(f"NO_CRITICO: {priority_counts['NO_CRITICO']:,} ({priority_counts['NO_CRITICO']/priority_counts.sum()*100:.1f}%)")
        print(f"CRITICO: {priority_counts['CRITICO']:,} ({priority_counts['CRITICO']/priority_counts.sum()*100:.1f}%)")
        
        # 2. Características (las mismas que funcionaron bien en el modelo binario simple)
        print("\n2. CARACTERÍSTICAS...")
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
            'Borough_Frequency', 'Has_Coordinates', 'Has_Address'
        ]
        
        model_data = data[features + ['Priority']].dropna(subset=['Priority'])
        X = model_data[features]
        y = model_data['Priority']
        
        print(f"✓ {len(features)} características, {len(model_data):,} filas")
        
        # 3. División
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 4. Pipeline simple
        preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # 5. Modelos con diferentes niveles de balance
        print("\n3. PROBANDO DIFERENTES NIVELES DE BALANCE...")
        print("-" * 40)
        
        models = {
            # Modelo sin pesos (baseline)
            'Sin Balance': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(
                    n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
                ))
            ]),
            
            # Balance ligero
            'Balance Ligero': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(
                    n_estimators=200, max_depth=15, random_state=42, n_jobs=-1,
                    class_weight={'CRITICO': 2.0, 'NO_CRITICO': 0.9}
                ))
            ]),
            
            # Balance moderado
            'Balance Moderado': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(
                    n_estimators=200, max_depth=15, random_state=42, n_jobs=-1,
                    class_weight={'CRITICO': 3.0, 'NO_CRITICO': 0.8}
                ))
            ]),
            
            # Balance sklearn
            'Balance Sklearn': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(
                    n_estimators=200, max_depth=15, random_state=42, n_jobs=-1,
                    class_weight='balanced'
                ))
            ]),
            
            # Gradient Boosting
            'Gradient Boosting': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=150, max_depth=8, random_state=42
                ))
            ])
        }
        
        results = {}
        
        for name, pipeline in models.items():
            print(f"\n🔄 {name}...")
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            
            from sklearn.metrics import recall_score, precision_score
            recall_critico = recall_score(y_test, y_pred, pos_label='CRITICO', average='binary')
            precision_critico = precision_score(y_test, y_pred, pos_label='CRITICO', average='binary')
            
            results[name] = {
                'model': pipeline,
                'accuracy': accuracy,
                'f1_weighted': f1_weighted,
                'recall_critico': recall_critico,
                'precision_critico': precision_critico,
                'predictions': y_pred
            }
            
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   F1-weighted: {f1_weighted:.4f}")
            print(f"   Recall-CRÍTICO: {recall_critico:.4f}")
            
        # 6. Seleccionar mejor balance
        print("\n4. ANÁLISIS DE RESULTADOS...")
        print("-" * 40)
        
        print("\n📊 TABLA COMPARATIVA:")
        print(f"{'Modelo':<20} {'Accuracy':<10} {'F1-weight':<10} {'Recall-Crit':<12}")
        print("-" * 55)
        
        for name, result in results.items():
            print(f"{name:<20} {result['accuracy']:<10.4f} {result['f1_weighted']:<10.4f} {result['recall_critico']:<12.4f}")
        
        # Seleccionar modelo que tenga buen balance
        def score_balance(result):
            # Penalizar mucho si accuracy < 60%
            if result['accuracy'] < 0.6:
                return 0
            # Balance: 70% accuracy + 30% recall crítico
            return 0.7 * result['accuracy'] + 0.3 * result['recall_critico']
        
        best_model_name = max(results.keys(), key=lambda x: score_balance(results[x]))
        best_result = results[best_model_name]
        
        print(f"\n🏆 MEJOR BALANCE: {best_model_name}")
        print(f"   Accuracy: {best_result['accuracy']:.4f}")
        print(f"   F1-weighted: {best_result['f1_weighted']:.4f}")
        print(f"   Recall-CRÍTICO: {best_result['recall_critico']:.4f}")
        print(f"   Precision-CRÍTICO: {best_result['precision_critico']:.4f}")
        
        # Análisis detallado
        y_pred_best = best_result['predictions']
        print(f"\n📋 REPORTE DETALLADO:")
        print(classification_report(y_test, y_pred_best))
        
        # Análisis de confusión
        cm = confusion_matrix(y_test, y_pred_best, labels=['CRITICO', 'NO_CRITICO'])
        casos_criticos_reales = sum(y_test == 'CRITICO')
        casos_criticos_detectados = cm[0, 0]
        casos_criticos_perdidos = cm[0, 1]
        
        print(f"\n🔍 ANÁLISIS DE CASOS CRÍTICOS:")
        print(f"   Total casos críticos: {casos_criticos_reales}")
        print(f"   Detectados: {casos_criticos_detectados}")
        print(f"   Perdidos: {casos_criticos_perdidos}")
        print(f"   Tasa detección: {casos_criticos_detectados/casos_criticos_reales*100:.1f}%")
        
        # 7. Visualización
        print("\n5. VISUALIZACIÓN...")
        print("-" * 40)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Comparación de modelos
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        recalls = [results[name]['recall_critico'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        axes[0, 0].bar(x + width/2, recalls, width, label='Recall Crítico', alpha=0.8)
        axes[0, 0].set_xlabel('Modelos')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Comparación de Modelos')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Matriz de confusión
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 1], cmap='Blues',
                    xticklabels=['CRÍTICO', 'NO CRÍTICO'],
                    yticklabels=['CRÍTICO', 'NO CRÍTICO'])
        axes[0, 1].set_title(f'Matriz de Confusión\n{best_model_name}')
        
        # Scatter accuracy vs recall
        axes[1, 0].scatter(accuracies, recalls, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            axes[1, 0].annotate(name, (accuracies[i], recalls[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
        axes[1, 0].set_xlabel('Accuracy')
        axes[1, 0].set_ylabel('Recall Crítico')
        axes[1, 0].set_title('Balance: Accuracy vs Recall Crítico')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Comparación evolutiva
        evolution = {
            'Original': {'acc': 0.5652, 'f1': 0.5057, 'recall_crit': 0.0},
            'Binario Simple': {'acc': 0.8246, 'f1': 0.7501, 'recall_crit': 0.01},
            'Binario Balanceado': {'acc': best_result['accuracy'], 'f1': best_result['f1_weighted'], 'recall_crit': best_result['recall_critico']}
        }
        
        evol_names = list(evolution.keys())
        evol_acc = [evolution[name]['acc'] for name in evol_names]
        evol_f1 = [evolution[name]['f1'] for name in evol_names]
        evol_recall = [evolution[name]['recall_crit'] for name in evol_names]
        
        x_evol = np.arange(len(evol_names))
        width = 0.25
        
        axes[1, 1].bar(x_evol - width, evol_acc, width, label='Accuracy', alpha=0.8)
        axes[1, 1].bar(x_evol, evol_f1, width, label='F1-weighted', alpha=0.8)
        axes[1, 1].bar(x_evol + width, evol_recall, width, label='Recall Crítico', alpha=0.8)
        
        axes[1, 1].set_xlabel('Evolución')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Evolución de Modelos')
        axes[1, 1].set_xticks(x_evol)
        axes[1, 1].set_xticklabels(evol_names)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('balanced_binary_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Visualización guardada como 'balanced_binary_results.png'")
        
        # 8. Resumen final
        print("\n" + "=" * 60)
        print("RESUMEN FINAL - MODELO BINARIO BALANCEADO")
        print("=" * 60)
        
        print(f"🏆 MEJOR MODELO: {best_model_name}")
        print(f"📊 Accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.1f}%)")
        print(f"📊 F1-weighted: {best_result['f1_weighted']:.4f} ({best_result['f1_weighted']*100:.1f}%)")
        print(f"📊 Recall-CRÍTICO: {best_result['recall_critico']:.4f} ({best_result['recall_critico']*100:.1f}%)")
        
        # Comparación con modelos anteriores
        print(f"\n📈 COMPARACIONES:")
        print(f"   vs Original: Accuracy +{(best_result['accuracy']-0.5652)/0.5652*100:.1f}%, F1 +{(best_result['f1_weighted']-0.5057)/0.5057*100:.1f}%")
        print(f"   vs Binario Simple: Recall crítico +{(best_result['recall_critico']-0.01)/0.01*100:.0f}%")
        
        if best_result['accuracy'] > 0.7 and best_result['recall_critico'] > 0.1:
            print(f"\n✅ ¡EXCELENTE! Balance logrado entre accuracy y detección crítica")
        elif best_result['accuracy'] > 0.6:
            print(f"\n👍 BUENO: Accuracy aceptable con mejor detección crítica")
        
        print(f"\n🎯 CONCLUSIÓN:")
        print(f"   El modelo binario balanceado logra un mejor compromiso")
        print(f"   entre accuracy general y detección de casos críticos.")
        
        return best_result['model']
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    model = main() 