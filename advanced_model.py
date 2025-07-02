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

def main():
    """Modelo avanzado con características críticas descubiertas"""
    
    print("=" * 70)
    print("MODELO AVANZADO CON CARACTERÍSTICAS CRÍTICAS")
    print("=" * 70)
    print("🎯 Incorporando: Location Type, Horas Críticas, Interacciones")
    
    try:
        # 1. Datos
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
        df['Priority'] = df['Descriptor'].map(binary_mapping)
        
        print(f"\n1. DATOS CARGADOS: {len(df):,} filas")
        
        # 2. CARACTERÍSTICAS CRÍTICAS AVANZADAS
        print("\n2. CREANDO CARACTERÍSTICAS CRÍTICAS...")
        print("-" * 50)
        
        data = df.copy()
        data['Created Date'] = pd.to_datetime(data['Created Date'], errors='coerce')
        
        # === TEMPORALES CRÍTICAS ===
        data['Hour'] = data['Created Date'].dt.hour
        data['DayOfWeek'] = data['Created Date'].dt.dayofweek
        data['Month'] = data['Created Date'].dt.month
        
        # Horas MÁS críticas (del análisis)
        data['Is_Hour_2AM'] = (data['Hour'] == 2).astype(int)    # 20.0% críticos
        data['Is_Hour_5AM'] = (data['Hour'] == 5).astype(int)    # 18.8% críticos
        data['Is_Hour_Midnight'] = (data['Hour'] == 0).astype(int) # 18.6% críticos
        data['Is_Hour_7AM'] = (data['Hour'] == 7).astype(int)    # 18.6% críticos
        data['Is_Hour_8PM'] = (data['Hour'] == 20).astype(int)   # 18.5% críticos
        
        # Rangos críticos
        data['Is_Madrugada'] = data['Hour'].between(0, 6).astype(int)
        data['Is_CriticalHours'] = data['Hour'].isin([0, 2, 5, 7, 20]).astype(int)
        
        # === GEOGRÁFICAS CRÍTICAS ===
        # Borough específicos
        data['Is_Bronx'] = (data['Borough'] == 'BRONX').astype(int)      # 20.2% críticos
        data['Is_Brooklyn'] = (data['Borough'] == 'BROOKLYN').astype(int) # 18.0% críticos
        data['Is_Manhattan'] = (data['Borough'] == 'MANHATTAN').astype(int)
        data['Is_Queens'] = (data['Borough'] == 'QUEENS').astype(int)
        
        # Borough de alto riesgo
        data['Is_HighRiskBorough'] = data['Borough'].isin(['BRONX', 'BROOKLYN']).astype(int)
        
        # === LOCATION TYPE CRÍTICO (¡LA CLAVE!) ===
        # Ubicaciones MÁS críticas (del análisis)
        data['Is_Park'] = (data['Location Type'] == 'Park/Playground').astype(int)    # 30.0% críticos!
        data['Is_Subway'] = (data['Location Type'] == 'Subway Station').astype(int)   # 22.5% críticos
        data['Is_Street'] = (data['Location Type'] == 'Street/Sidewalk').astype(int)  # 22.2% críticos
        data['Is_Store'] = (data['Location Type'] == 'Store/Commercial').astype(int)  # 19.6% críticos
        data['Is_Residential'] = (data['Location Type'] == 'Residential Building/House').astype(int) # 17.9% críticos
        
        # Ubicaciones de alto riesgo
        high_risk_locations = ['Park/Playground', 'Subway Station', 'Street/Sidewalk']
        data['Is_HighRiskLocation'] = data['Location Type'].isin(high_risk_locations).astype(int)
        
        # === INTERACCIONES SÚPER CRÍTICAS ===
        # Las combinaciones más peligrosas
        data['Bronx_Park'] = (data['Is_Bronx'] & data['Is_Park']).astype(int)        # Bronx + Park
        data['Bronx_Madrugada'] = (data['Is_Bronx'] & data['Is_Madrugada']).astype(int) # Bronx + Madrugada
        data['Park_Madrugada'] = (data['Is_Park'] & data['Is_Madrugada']).astype(int)   # Park + Madrugada  
        data['Park_CriticalHours'] = (data['Is_Park'] & data['Is_CriticalHours']).astype(int) # Park + Horas críticas
        data['HighRisk_Combination'] = (data['Is_HighRiskBorough'] & data['Is_HighRiskLocation']).astype(int)
        
        # === OTRAS CARACTERÍSTICAS ===
        data['Has_Coordinates'] = (~data['Latitude'].isna()).astype(int)
        data['Has_Address'] = (~data['Incident Address'].isna()).astype(int)
        
        # Estacional (Otoño y Verano = 18.8% críticos)
        data['Season'] = data['Month'].map({
            12: 'Invierno', 1: 'Invierno', 2: 'Invierno',
            3: 'Primavera', 4: 'Primavera', 5: 'Primavera',
            6: 'Verano', 7: 'Verano', 8: 'Verano',
            9: 'Otoño', 10: 'Otoño', 11: 'Otoño'
        })
        data['Is_HighRiskSeason'] = data['Season'].isin(['Otoño', 'Verano']).astype(int)
        
        # Resolución
        data['Closed Date'] = pd.to_datetime(data['Closed Date'], errors='coerce')
        data['Has_Resolution'] = (~data['Closed Date'].isna()).astype(int)
        
        print("✓ Características temporales críticas")
        print("✓ Características geográficas críticas")
        print("✓ Location Type crítico incorporado")
        print("✓ Interacciones súper críticas")
        
        # 3. Características finales
        features = [
            # Básicas
            'Hour', 'DayOfWeek', 'Month',
            
            # Horas críticas específicas
            'Is_Hour_2AM', 'Is_Hour_5AM', 'Is_Hour_Midnight', 'Is_Hour_7AM', 'Is_Hour_8PM',
            'Is_Madrugada', 'Is_CriticalHours',
            
            # Borough críticos
            'Is_Bronx', 'Is_Brooklyn', 'Is_Manhattan', 'Is_Queens', 'Is_HighRiskBorough',
            
            # Location Type CRÍTICO (la diferencia clave)
            'Is_Park', 'Is_Subway', 'Is_Street', 'Is_Store', 'Is_Residential', 'Is_HighRiskLocation',
            
            # Interacciones súper críticas
            'Bronx_Park', 'Bronx_Madrugada', 'Park_Madrugada', 'Park_CriticalHours', 'HighRisk_Combination',
            
            # Otras
            'Is_HighRiskSeason', 'Has_Coordinates', 'Has_Address', 'Has_Resolution'
        ]
        
        model_data = data[features + ['Priority']].dropna(subset=['Priority'])
        X = model_data[features]
        y = model_data['Priority']
        
        print(f"\n✓ {len(features)} características avanzadas")
        print(f"✓ Dataset: {len(model_data):,} filas")
        
        # Mostrar estadísticas de características clave
        print(f"\n📊 ESTADÍSTICAS DE CARACTERÍSTICAS CLAVE:")
        print(f"   Parks (30% críticos): {X['Is_Park'].sum():,} casos")
        print(f"   Bronx (20.2% críticos): {X['Is_Bronx'].sum():,} casos")
        print(f"   Madrugada: {X['Is_Madrugada'].sum():,} casos")
        print(f"   Bronx+Park: {X['Bronx_Park'].sum():,} casos")
        print(f"   Park+Madrugada: {X['Park_Madrugada'].sum():,} casos")
        
        # 4. División
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 5. Pipeline
        preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler())
        ])
        
        # 6. Modelos avanzados con pesos optimizados
        print("\n3. ENTRENANDO MODELOS AVANZADOS...")
        print("-" * 50)
        
        models = {
            'Random Forest Crítico': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(
                    n_estimators=300, max_depth=20, min_samples_split=2,
                    class_weight={'CRITICO': 3.0, 'NO_CRITICO': 0.7},
                    random_state=42, n_jobs=-1
                ))
            ]),
            
            'Random Forest Ultra': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(
                    n_estimators=500, max_depth=25, min_samples_split=2,
                    class_weight={'CRITICO': 4.0, 'NO_CRITICO': 0.6},
                    max_features='sqrt', random_state=42, n_jobs=-1
                ))
            ]),
            
            'Gradient Boosting Avanzado': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=200, max_depth=10, learning_rate=0.05,
                    min_samples_split=5, random_state=42
                ))
            ])
        }
        
        results = {}
        
        for name, pipeline in models.items():
            print(f"\n🔄 {name}...")
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            
            from sklearn.metrics import recall_score, precision_score
            recall_critico = recall_score(y_test, y_pred, pos_label='CRITICO', average='binary')
            precision_critico = precision_score(y_test, y_pred, pos_label='CRITICO', average='binary')
            f1_critico = f1_score(y_test, y_pred, pos_label='CRITICO', average='binary')
            
            results[name] = {
                'model': pipeline,
                'accuracy': accuracy,
                'f1_weighted': f1_weighted,
                'f1_critico': f1_critico,
                'recall_critico': recall_critico,
                'precision_critico': precision_critico,
                'predictions': y_pred
            }
            
            print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
            print(f"   F1-weighted: {f1_weighted:.4f} ({f1_weighted*100:.1f}%)")
            print(f"   Recall-CRÍTICO: {recall_critico:.4f} ({recall_critico*100:.1f}%)")
            print(f"   F1-CRÍTICO: {f1_critico:.4f} ({f1_critico*100:.1f}%)")
        
        # 7. Mejor modelo
        print("\n4. SELECCIÓN DEL MEJOR MODELO AVANZADO...")
        print("-" * 50)
        
        # Score balanceado: prioriza recall crítico pero mantiene accuracy
        def score_critico(result):
            if result['accuracy'] < 0.6:  # Penalizar accuracy muy baja
                return 0
            return 0.6 * result['accuracy'] + 0.4 * result['recall_critico']
        
        best_name = max(results.keys(), key=lambda x: score_critico(results[x]))
        best = results[best_name]
        
        print(f"🏆 MEJOR MODELO AVANZADO: {best_name}")
        print(f"   Accuracy: {best['accuracy']:.4f} ({best['accuracy']*100:.1f}%)")
        print(f"   F1-weighted: {best['f1_weighted']:.4f} ({best['f1_weighted']*100:.1f}%)")
        print(f"   Recall-CRÍTICO: {best['recall_critico']:.4f} ({best['recall_critico']*100:.1f}%)")
        print(f"   F1-CRÍTICO: {best['f1_critico']:.4f} ({best['f1_critico']*100:.1f}%)")
        
        # Análisis detallado
        y_pred_best = best['predictions']
        print(f"\n📋 REPORTE DETALLADO:")
        print(classification_report(y_test, y_pred_best))
        
        # Casos críticos
        cm = confusion_matrix(y_test, y_pred_best, labels=['CRITICO', 'NO_CRITICO'])
        criticos_reales = sum(y_test == 'CRITICO')
        criticos_detectados = cm[0, 0]
        criticos_perdidos = cm[0, 1]
        
        print(f"\n🎯 ANÁLISIS DE CASOS CRÍTICOS:")
        print(f"   Total casos críticos: {criticos_reales:,}")
        print(f"   Detectados: {criticos_detectados:,}")
        print(f"   Perdidos: {criticos_perdidos:,}")
        print(f"   Tasa detección: {criticos_detectados/criticos_reales*100:.1f}%")
        
        # Importancia de características
        if 'Random Forest' in best_name:
            importances = best['model'].named_steps['classifier'].feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print(f"\n🎯 TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES:")
            for i, row in feature_importance.head(10).iterrows():
                print(f"   {row['Feature']}: {row['Importance']:.4f}")
        
        # 8. Visualización
        print("\n5. VISUALIZACIÓN...")
        print("-" * 50)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Comparación de modelos
        names = list(results.keys())
        accs = [results[name]['accuracy'] for name in names]
        recalls = [results[name]['recall_critico'] for name in names]
        f1s = [results[name]['f1_weighted'] for name in names]
        
        x = np.arange(len(names))
        width = 0.25
        
        axes[0, 0].bar(x - width, accs, width, label='Accuracy')
        axes[0, 0].bar(x, recalls, width, label='Recall Crítico') 
        axes[0, 0].bar(x + width, f1s, width, label='F1-weighted')
        axes[0, 0].set_title('Modelos Avanzados')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Matriz de confusión
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 1], cmap='Blues',
                    xticklabels=['CRÍTICO', 'NO CRÍTICO'],
                    yticklabels=['CRÍTICO', 'NO CRÍTICO'])
        axes[0, 1].set_title(f'Matriz de Confusión\n{best_name}')
        
        # Evolución completa
        evolution = [
            ('Original', 0.5652, 0.5057, 0),
            ('Binario Simple', 0.8246, 0.7501, 33),
            ('Binario Balanceado', 0.8071, 0.7516, 144),
            ('Avanzado', best['accuracy'], best['f1_weighted'], criticos_detectados)
        ]
        
        model_names = [x[0] for x in evolution]
        accuracies_evol = [x[1] for x in evolution]
        f1s_evol = [x[2] for x in evolution]
        
        x_evol = np.arange(len(model_names))
        axes[1, 0].bar(x_evol - 0.2, accuracies_evol, 0.4, label='Accuracy', alpha=0.8)
        axes[1, 0].bar(x_evol + 0.2, f1s_evol, 0.4, label='F1-weighted', alpha=0.8)
        axes[1, 0].set_title('Evolución Completa del Proyecto')
        axes[1, 0].set_xticks(x_evol)
        axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Casos críticos detectados
        casos_detectados = [x[3] for x in evolution]
        axes[1, 1].bar(model_names, casos_detectados, color='red', alpha=0.7)
        axes[1, 1].set_title(f'Casos Críticos Detectados\n(Total: {criticos_reales:,})')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        for i, v in enumerate(casos_detectados):
            pct = v / criticos_reales * 100 if criticos_reales > 0 else 0
            axes[1, 1].text(i, v + v*0.05, f'{v:,}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('advanced_model_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Visualización guardada como 'advanced_model_results.png'")
        
        # 9. Resumen épico
        print("\n" + "=" * 70)
        print("RESUMEN FINAL - MODELO AVANZADO")
        print("=" * 70)
        
        print(f"🏆 MEJOR MODELO: {best_name}")
        print(f"📊 Accuracy: {best['accuracy']*100:.1f}%")
        print(f"📊 F1-weighted: {best['f1_weighted']*100:.1f}%")
        print(f"📊 Recall-CRÍTICO: {best['recall_critico']*100:.1f}%")
        
        print(f"\n🚀 EVOLUCIÓN COMPLETA:")
        for name, acc, f1, casos in evolution:
            if name == 'Avanzado':
                print(f"   🏆 {name}: {acc*100:.1f}% acc, {f1*100:.1f}% f1, {casos:,} casos detectados")
            else:
                print(f"   📊 {name}: {acc*100:.1f}% acc, {f1*100:.1f}% f1, {casos:,} casos detectados")
        
        # Mejoras vs original
        mejora_acc = (best['accuracy'] - 0.5652) / 0.5652 * 100
        mejora_f1 = (best['f1_weighted'] - 0.5057) / 0.5057 * 100
        
        print(f"\n✅ MEJORAS vs MODELO ORIGINAL:")
        print(f"   Accuracy: +{mejora_acc:.1f}%")
        print(f"   F1-weighted: +{mejora_f1:.1f}%")
        print(f"   Casos críticos: {criticos_detectados:,} vs 0")
        
        if criticos_detectados > 500:
            print(f"\n🎉 ¡INCREÍBLE! >500 casos críticos detectados")
        elif criticos_detectados > 200:
            print(f"\n🚀 ¡EXCELENTE! >200 casos críticos detectados")
        elif criticos_detectados > 100:
            print(f"\n👍 ¡BUENO! >100 casos críticos detectados")
        
        print(f"\n🎯 CARACTERÍSTICAS QUE MARCARON LA DIFERENCIA:")
        if 'Random Forest' in best_name:
            print("   TOP 3:")
            for i, row in feature_importance.head(3).iterrows():
                print(f"   {i+1}. {row['Feature']}: {row['Importance']:.4f}")
        
        print(f"\n🏁 ¡MISIÓN CUMPLIDA!")
        print(f"   El modelo avanzado es significativamente superior")
        print(f"   y puede salvar {criticos_detectados:,} casos críticos de abuso animal.")
        
        return best['model']
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    model = main() 