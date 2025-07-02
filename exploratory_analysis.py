import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para español
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('default')

def load_and_explore_data():
    """Cargar y explorar el dataset de abuso animal de NYC"""
    
    print("=" * 60)
    print("ANÁLISIS EXPLORATORIO - DATASET NYC ANIMAL ABUSE")
    print("=" * 60)
    
    # Cargar datos
    print("\n1. CARGANDO DATOS...")
    try:
        df = pd.read_csv('Animal_Abuse.csv/Animal_Abuse.csv')
        print(f"✓ Dataset cargado exitosamente!")
        print(f"  - Filas: {df.shape[0]:,}")
        print(f"  - Columnas: {df.shape[1]}")
    except Exception as e:
        print(f"✗ Error al cargar datos: {e}")
        return None
    
    # Información básica del dataset
    print("\n2. INFORMACIÓN BÁSICA DEL DATASET")
    print("-" * 40)
    print(f"Forma del dataset: {df.shape}")
    print(f"Memoria utilizada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Mostrar primeras filas
    print("\n3. PRIMERAS 5 FILAS:")
    print("-" * 40)
    print(df.head().to_string())
    
    # Información de columnas
    print("\n4. INFORMACIÓN DE COLUMNAS:")
    print("-" * 40)
    print(df.info())
    
    # Estadísticas descriptivas para columnas numéricas
    print("\n5. ESTADÍSTICAS DESCRIPTIVAS (COLUMNAS NUMÉRICAS):")
    print("-" * 50)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
    else:
        print("No hay columnas numéricas en el dataset")
    
    # Análisis de valores faltantes
    print("\n6. ANÁLISIS DE VALORES FALTANTES:")
    print("-" * 40)
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Columna': missing_data.index,
        'Valores_Faltantes': missing_data.values,
        'Porcentaje': missing_percent.values
    })
    missing_df = missing_df[missing_df['Valores_Faltantes'] > 0].sort_values('Porcentaje', ascending=False)
    print(missing_df.to_string(index=False))
    
    # Análisis de la variable objetivo potencial (Descriptor)
    print("\n7. ANÁLISIS DE LA VARIABLE OBJETIVO - 'Descriptor':")
    print("-" * 50)
    if 'Descriptor' in df.columns:
        descriptor_counts = df['Descriptor'].value_counts()
        descriptor_percent = (descriptor_counts / len(df)) * 100
        
        print("Distribución de tipos de abuso:")
        for desc, count in descriptor_counts.items():
            percent = descriptor_percent[desc]
            print(f"  - {desc}: {count:,} ({percent:.2f}%)")
        
        # Guardar info para visualización
        return df, descriptor_counts
    else:
        print("✗ Columna 'Descriptor' no encontrada")
        return df, None

def create_visualizations(df, descriptor_counts):
    """Crear visualizaciones exploratorias"""
    
    print("\n8. CREANDO VISUALIZACIONES...")
    print("-" * 30)
    
    # Configurar el estilo
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Distribución de tipos de abuso
    if descriptor_counts is not None:
        plt.subplot(2, 3, 1)
        colors = plt.cm.Set3(np.linspace(0, 1, len(descriptor_counts)))
        bars = plt.bar(descriptor_counts.index, descriptor_counts.values, color=colors)
        plt.title('Distribución de Tipos de Abuso Animal', fontsize=14, fontweight='bold')
        plt.xlabel('Tipo de Abuso')
        plt.ylabel('Número de Casos')
        plt.xticks(rotation=45)
        
        # Añadir valores encima de las barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom')
    
    # 2. Distribución por Borough
    if 'Borough' in df.columns:
        plt.subplot(2, 3, 2)
        borough_counts = df['Borough'].value_counts()
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(borough_counts)))
        plt.pie(borough_counts.values, labels=borough_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title('Distribución por Borough', fontsize=14, fontweight='bold')
    
    # 3. Análisis temporal (si existe Created Date)
    if 'Created Date' in df.columns:
        plt.subplot(2, 3, 3)
        try:
            df['Created Date'] = pd.to_datetime(df['Created Date'])
            df['Month'] = df['Created Date'].dt.month
            monthly_counts = df['Month'].value_counts().sort_index()
            months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                     'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dec']
            plt.plot(monthly_counts.index, monthly_counts.values, 
                    marker='o', linewidth=2, markersize=6)
            plt.title('Casos por Mes', fontsize=14, fontweight='bold')
            plt.xlabel('Mes')
            plt.ylabel('Número de Casos')
            plt.xticks(range(1, 13), months, rotation=45)
            plt.grid(True, alpha=0.3)
        except:
            plt.text(0.5, 0.5, 'Error procesando fechas', ha='center', va='center')
    
    # 4. Status de los casos
    if 'Status' in df.columns:
        plt.subplot(2, 3, 4)
        status_counts = df['Status'].value_counts()
        colors = ['#2ecc71' if status == 'Closed' else '#e74c3c' for status in status_counts.index]
        bars = plt.bar(status_counts.index, status_counts.values, color=colors)
        plt.title('Estado de los Casos', fontsize=14, fontweight='bold')
        plt.xlabel('Estado')
        plt.ylabel('Número de Casos')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom')
    
    # 5. Mapa de calor de valores faltantes
    plt.subplot(2, 3, 5)
    missing_data = df.isnull().sum()
    top_missing = missing_data[missing_data > 0].head(10)
    if len(top_missing) > 0:
        colors = plt.cm.Reds(np.linspace(0.3, 1, len(top_missing)))
        bars = plt.barh(range(len(top_missing)), top_missing.values, color=colors)
        plt.yticks(range(len(top_missing)), top_missing.index, fontsize=10)
        plt.xlabel('Número de Valores Faltantes')
        plt.title('Top 10 Columnas con Valores Faltantes', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
    
    # 6. Análisis de coordenadas (si existen)
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        plt.subplot(2, 3, 6)
        # Filtrar valores válidos de coordenadas
        valid_coords = df.dropna(subset=['Latitude', 'Longitude'])
        if len(valid_coords) > 0:
            plt.scatter(valid_coords['Longitude'], valid_coords['Latitude'], 
                       alpha=0.6, s=1, c='red')
            plt.title('Distribución Geográfica de Casos', fontsize=14, fontweight='bold')
            plt.xlabel('Longitud')
            plt.ylabel('Latitud')
        else:
            plt.text(0.5, 0.5, 'No hay coordenadas válidas', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('exploratory_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Visualizaciones guardadas como 'exploratory_analysis.png'")

def analyze_prediction_target(df):
    """Analizar la variable objetivo para predicción"""
    
    print("\n9. ANÁLISIS PARA PREDICCIÓN:")
    print("-" * 30)
    
    if 'Descriptor' not in df.columns:
        print("✗ No se encontró la columna 'Descriptor' para usar como variable objetivo")
        return
    
    # Análisis de balance de clases
    print("Balance de clases en 'Descriptor':")
    descriptor_counts = df['Descriptor'].value_counts()
    total = len(df)
    
    for desc, count in descriptor_counts.items():
        percentage = (count / total) * 100
        print(f"  - {desc}: {count:,} casos ({percentage:.2f}%)")
    
    # Verificar si hay desbalance significativo
    max_class = descriptor_counts.max()
    min_class = descriptor_counts.min()
    imbalance_ratio = max_class / min_class
    
    print(f"\nRatio de desbalance: {imbalance_ratio:.2f}")
    if imbalance_ratio > 3:
        print("⚠️  ADVERTENCIA: Hay desbalance significativo en las clases")
        print("   Considerar técnicas de balanceamiento (SMOTE, undersampling, etc.)")
    else:
        print("✓ Las clases están relativamente balanceadas")
    
    # Características potenciales para predicción
    print("\nCaracterísticas potenciales para predicción:")
    potential_features = []
    
    for col in df.columns:
        if col != 'Descriptor':  # Excluir la variable objetivo
            non_null_count = df[col].count()
            null_percentage = (len(df) - non_null_count) / len(df) * 100
            
            if null_percentage < 50:  # Solo considerar columnas con menos del 50% de valores faltantes
                potential_features.append((col, null_percentage))
    
    # Ordenar por porcentaje de valores faltantes
    potential_features.sort(key=lambda x: x[1])
    
    print(f"Se identificaron {len(potential_features)} características potenciales:")
    for i, (feature, null_pct) in enumerate(potential_features[:15]):  # Mostrar top 15
        print(f"  {i+1:2d}. {feature} (valores faltantes: {null_pct:.1f}%)")
    
    return potential_features

def main():
    """Función principal"""
    try:
        # Cargar y explorar datos
        result = load_and_explore_data()
        if result is None:
            return
        
        df, descriptor_counts = result
        
        # Crear visualizaciones
        create_visualizations(df, descriptor_counts)
        
        # Analizar variable objetivo
        potential_features = analyze_prediction_target(df)
        
        print("\n" + "=" * 60)
        print("RESUMEN DEL ANÁLISIS EXPLORATORIO")
        print("=" * 60)
        print(f"✓ Dataset cargado: {df.shape[0]:,} filas, {df.shape[1]} columnas")
        print(f"✓ Variable objetivo identificada: 'Descriptor' con {df['Descriptor'].nunique()} clases")
        print(f"✓ Características potenciales: {len(potential_features) if potential_features else 0}")
        print("✓ Visualizaciones creadas y guardadas")
        print("\nPróximos pasos sugeridos:")
        print("1. Preprocesamiento de datos")
        print("2. Ingeniería de características")
        print("3. Selección de modelo")
        print("4. Entrenamiento y evaluación")
        
    except Exception as e:
        print(f"✗ Error en el análisis: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 