import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def analyze_critical_cases():
    """Análisis profundo: ¿Qué hace crítico a un caso?"""
    
    print("=" * 70)
    print("ANÁLISIS PROFUNDO: ¿QUÉ CARACTERIZA A LOS CASOS CRÍTICOS?")
    print("=" * 70)
    
    # Cargar datos
    df = pd.read_csv('Animal_Abuse.csv/Animal_Abuse.csv')
    
    # Clasificar
    binary_mapping = {
        'Tortured': 'CRITICO',
        'Chained': 'CRITICO',
        'Neglected': 'NO_CRITICO',
        'No Shelter': 'NO_CRITICO',
        'In Car': 'NO_CRITICO',
        'Noise, Barking Dog (NR5)': 'NO_CRITICO',
        'Other (complaint details)': 'NO_CRITICO'
    }
    
    df['Priority'] = df['Descriptor'].map(binary_mapping)
    criticos = df[df['Priority'] == 'CRITICO'].copy()
    no_criticos = df[df['Priority'] == 'NO_CRITICO'].copy()
    
    print(f"📊 DATOS:")
    print(f"   Casos críticos: {len(criticos):,}")
    print(f"   Casos no críticos: {len(no_criticos):,}")
    
    # 1. ANÁLISIS TEMPORAL DETALLADO
    print("\n" + "="*50)
    print("1. ANÁLISIS TEMPORAL DETALLADO")
    print("="*50)
    
    df['Created Date'] = pd.to_datetime(df['Created Date'], errors='coerce')
    df['Hour'] = df['Created Date'].dt.hour
    df['DayOfWeek'] = df['Created Date'].dt.dayofweek
    df['Month'] = df['Created Date'].dt.month
    
    # Horas más críticas
    print("\n🕐 DISTRIBUCIÓN POR HORA:")
    criticos['Created Date'] = pd.to_datetime(criticos['Created Date'], errors='coerce')
    no_criticos['Created Date'] = pd.to_datetime(no_criticos['Created Date'], errors='coerce')
    
    criticos['Hour'] = criticos['Created Date'].dt.hour
    no_criticos['Hour'] = no_criticos['Created Date'].dt.hour
    
    hour_criticos = criticos['Hour'].value_counts().sort_index()
    hour_no_criticos = no_criticos['Hour'].value_counts().sort_index()
    
    # Ratio crítico/no_crítico por hora
    hour_ratio = (hour_criticos / (hour_criticos + hour_no_criticos) * 100).fillna(0)
    
    print("Horas con MAYOR proporción de casos críticos:")
    top_hours = hour_ratio.nlargest(5)
    for hour, ratio in top_hours.items():
        total_hour = hour_criticos.get(hour, 0) + hour_no_criticos.get(hour, 0)
        print(f"   {hour:2d}:00 - {ratio:.1f}% críticos ({hour_criticos.get(hour, 0)} de {total_hour} casos)")
    
    # 2. ANÁLISIS GEOGRÁFICO DETALLADO
    print("\n" + "="*50)
    print("2. ANÁLISIS GEOGRÁFICO DETALLADO")
    print("="*50)
    
    # Borough analysis
    print("\n🏙️ POR BOROUGH:")
    borough_analysis = df.groupby(['Borough', 'Priority']).size().unstack(fill_value=0)
    borough_analysis['Total'] = borough_analysis.sum(axis=1)
    borough_analysis['Pct_Critico'] = (borough_analysis['CRITICO'] / borough_analysis['Total'] * 100).round(1)
    
    print(borough_analysis[['CRITICO', 'NO_CRITICO', 'Total', 'Pct_Critico']].sort_values('Pct_Critico', ascending=False))
    
    # Location Type analysis
    print("\n📍 POR TIPO DE UBICACIÓN:")
    location_analysis = df.groupby(['Location Type', 'Priority']).size().unstack(fill_value=0)
    location_analysis['Total'] = location_analysis.sum(axis=1)
    location_analysis['Pct_Critico'] = (location_analysis['CRITICO'] / location_analysis['Total'] * 100).round(1)
    
    # Solo mostrar ubicaciones con suficientes casos
    location_filtered = location_analysis[location_analysis['Total'] >= 100].sort_values('Pct_Critico', ascending=False)
    print("\nUbicaciones con >100 casos, ordenadas por % críticos:")
    print(location_filtered[['CRITICO', 'NO_CRITICO', 'Total', 'Pct_Critico']].head(10))
    
    # 3. ANÁLISIS DE COORDENADAS
    print("\n" + "="*50)
    print("3. ANÁLISIS DE COORDENADAS")
    print("="*50)
    
    coords_criticos = criticos.dropna(subset=['Latitude', 'Longitude'])
    coords_no_criticos = no_criticos.dropna(subset=['Latitude', 'Longitude'])
    
    print(f"Casos críticos con coordenadas: {len(coords_criticos):,}/{len(criticos):,} ({len(coords_criticos)/len(criticos)*100:.1f}%)")
    print(f"Casos no críticos con coordenadas: {len(coords_no_criticos):,}/{len(no_criticos):,} ({len(coords_no_criticos)/len(no_criticos)*100:.1f}%)")
    
    if len(coords_criticos) > 0:
        print(f"\n📊 ESTADÍSTICAS DE UBICACIÓN:")
        print(f"Críticos - Lat: {coords_criticos['Latitude'].mean():.4f}±{coords_criticos['Latitude'].std():.4f}")
        print(f"Críticos - Lon: {coords_criticos['Longitude'].mean():.4f}±{coords_criticos['Longitude'].std():.4f}")
        print(f"No críticos - Lat: {coords_no_criticos['Latitude'].mean():.4f}±{coords_no_criticos['Latitude'].std():.4f}")
        print(f"No críticos - Lon: {coords_no_criticos['Longitude'].mean():.4f}±{coords_no_criticos['Longitude'].std():.4f}")
    
    # 4. ANÁLISIS DE AGENCIAS
    print("\n" + "="*50)
    print("4. ANÁLISIS DE AGENCIAS")
    print("="*50)
    
    agency_analysis = df.groupby(['Agency', 'Priority']).size().unstack(fill_value=0)
    agency_analysis['Total'] = agency_analysis.sum(axis=1)
    agency_analysis['Pct_Critico'] = (agency_analysis['CRITICO'] / agency_analysis['Total'] * 100).round(1)
    
    print("📋 POR AGENCIA:")
    print(agency_analysis[['CRITICO', 'NO_CRITICO', 'Total', 'Pct_Critico']].sort_values('Total', ascending=False))
    
    # 5. ANÁLISIS DE RESOLUCIÓN
    print("\n" + "="*50)
    print("5. ANÁLISIS DE RESOLUCIÓN")
    print("="*50)
    
    df['Closed Date'] = pd.to_datetime(df['Closed Date'], errors='coerce')
    df['Has_Resolution'] = ~df['Closed Date'].isna()
    df['Resolution_Hours'] = (df['Closed Date'] - df['Created Date']).dt.total_seconds() / 3600
    
    print("🔄 TASAS DE RESOLUCIÓN:")
    resolution_by_priority = df.groupby('Priority')['Has_Resolution'].agg(['count', 'sum', 'mean'])
    resolution_by_priority.columns = ['Total_Cases', 'Resolved_Cases', 'Resolution_Rate']
    resolution_by_priority['Resolution_Rate'] = (resolution_by_priority['Resolution_Rate'] * 100).round(1)
    print(resolution_by_priority)
    
    print("\n⏱️ TIEMPO DE RESOLUCIÓN (horas):")
    resolution_times = df[df['Has_Resolution']].groupby('Priority')['Resolution_Hours'].describe()
    print(resolution_times[['count', 'mean', 'std', '50%', '75%', 'max']])
    
    # 6. ANÁLISIS DE DATOS FALTANTES
    print("\n" + "="*50)
    print("6. ANÁLISIS DE DATOS FALTANTES")
    print("="*50)
    
    missing_analysis = []
    key_columns = ['Incident Address', 'Incident Zip', 'Latitude', 'Longitude', 'Location Type', 'Closed Date']
    
    for col in key_columns:
        criticos_missing = criticos[col].isna().sum()
        no_criticos_missing = no_criticos[col].isna().sum()
        criticos_pct = (criticos_missing / len(criticos) * 100)
        no_criticos_pct = (no_criticos_missing / len(no_criticos) * 100)
        
        missing_analysis.append({
            'Column': col,
            'Criticos_Missing': criticos_missing,
            'Criticos_Pct': criticos_pct,
            'No_Criticos_Missing': no_criticos_missing,
            'No_Criticos_Pct': no_criticos_pct,
            'Difference': criticos_pct - no_criticos_pct
        })
    
    missing_df = pd.DataFrame(missing_analysis)
    print("📊 DATOS FALTANTES POR TIPO:")
    print(missing_df.round(2))
    
    # 7. CARACTERÍSTICAS POTENCIALES NO EXPLORADAS
    print("\n" + "="*50)
    print("7. CARACTERÍSTICAS NO EXPLORADAS")
    print("="*50)
    
    print("🔍 CAMPOS CON INFORMACIÓN TEXTUAL:")
    text_fields = ['Descriptor', 'Location Type', 'Status', 'Resolution Description', 'Community Board']
    
    for field in text_fields:
        if field in df.columns:
            unique_values = df[field].nunique()
            missing_pct = df[field].isna().sum() / len(df) * 100
            print(f"   {field}: {unique_values} valores únicos, {missing_pct:.1f}% faltantes")
            
            if field == 'Location Type':
                print("      Top ubicaciones críticas:")
                location_crit = criticos[field].value_counts().head(3)
                for loc, count in location_crit.items():
                    print(f"        - {loc}: {count} casos")
    
    # 8. ANÁLISIS DE PATRONES ESTACIONALES
    print("\n" + "="*50)
    print("8. ANÁLISIS ESTACIONAL")
    print("="*50)
    
    df['Month'] = df['Created Date'].dt.month
    df['Season'] = df['Month'].map({
        12: 'Invierno', 1: 'Invierno', 2: 'Invierno',
        3: 'Primavera', 4: 'Primavera', 5: 'Primavera',
        6: 'Verano', 7: 'Verano', 8: 'Verano',
        9: 'Otoño', 10: 'Otoño', 11: 'Otoño'
    })
    
    seasonal_analysis = df.groupby(['Season', 'Priority']).size().unstack(fill_value=0)
    seasonal_analysis['Total'] = seasonal_analysis.sum(axis=1)
    seasonal_analysis['Pct_Critico'] = (seasonal_analysis['CRITICO'] / seasonal_analysis['Total'] * 100).round(1)
    
    print("🌞 POR ESTACIÓN:")
    print(seasonal_analysis[['CRITICO', 'NO_CRITICO', 'Total', 'Pct_Critico']])
    
    # 9. VISUALIZACIÓN
    print("\n" + "="*50)
    print("9. CREANDO VISUALIZACIONES")
    print("="*50)
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # 1. Distribución por hora
    hour_comparison = pd.DataFrame({
        'Críticos': hour_criticos,
        'No Críticos': hour_no_criticos
    }).fillna(0)
    
    hour_comparison.plot(kind='bar', ax=axes[0, 0], width=0.8)
    axes[0, 0].set_title('Distribución por Hora del Día')
    axes[0, 0].set_xlabel('Hora')
    axes[0, 0].set_ylabel('Número de Casos')
    axes[0, 0].legend()
    
    # 2. Ratio crítico por hora
    hour_ratio.plot(kind='bar', ax=axes[0, 1], color='red', alpha=0.7)
    axes[0, 1].set_title('% de Casos Críticos por Hora')
    axes[0, 1].set_xlabel('Hora')
    axes[0, 1].set_ylabel('% Críticos')
    
    # 3. Por borough
    borough_analysis[['CRITICO', 'NO_CRITICO']].plot(kind='bar', ax=axes[1, 0], stacked=True)
    axes[1, 0].set_title('Casos por Borough')
    axes[1, 0].set_xlabel('Borough')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. % críticos por borough
    borough_analysis['Pct_Critico'].plot(kind='bar', ax=axes[1, 1], color='orange', alpha=0.7)
    axes[1, 1].set_title('% Casos Críticos por Borough')
    axes[1, 1].set_xlabel('Borough')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 5. Por estación
    seasonal_analysis[['CRITICO', 'NO_CRITICO']].plot(kind='bar', ax=axes[2, 0])
    axes[2, 0].set_title('Casos por Estación')
    axes[2, 0].set_xlabel('Estación')
    axes[2, 0].tick_params(axis='x', rotation=0)
    
    # 6. Top ubicaciones críticas
    if len(location_filtered) > 0:
        top_locations = location_filtered.head(8)
        top_locations['Pct_Critico'].plot(kind='barh', ax=axes[2, 1], color='red', alpha=0.7)
        axes[2, 1].set_title('% Críticos por Tipo de Ubicación')
        axes[2, 1].set_xlabel('% Casos Críticos')
    
    plt.tight_layout()
    plt.savefig('deep_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Visualizaciones guardadas como 'deep_analysis_results.png'")
    
    # 10. RECOMENDACIONES PARA MEJORAR EL MODELO
    print("\n" + "="*70)
    print("RECOMENDACIONES PARA MEJORAR EL MODELO")
    print("="*70)
    
    print("\n🎯 CARACTERÍSTICAS NUEVAS A PROBAR:")
    print("   1. Location Type específico (algunas ubicaciones tienen >30% críticos)")
    print("   2. Combinaciones hora-borough (interacciones)")
    print("   3. Características estacionales más específicas")
    print("   4. Densidad de casos por zona geográfica")
    print("   5. Análisis de texto en Resolution Description")
    
    print("\n🔧 TÉCNICAS AVANZADAS:")
    print("   1. Feature engineering con interacciones")
    print("   2. Algoritmos más sofisticados (XGBoost, LightGBM)")
    print("   3. Ensemble methods")
    print("   4. Análisis de clustering geográfico")
    print("   5. Análisis de series temporales")
    
    print("\n⚠️ PROBLEMAS IDENTIFICADOS:")
    print("   1. Datos faltantes significativos en coordenadas")
    print("   2. Desbalance extremo (17% vs 83%)")
    print("   3. Características actuales muy básicas")
    print("   4. No aprovechamos información textual")
    
    return df

if __name__ == "__main__":
    data = analyze_critical_cases() 