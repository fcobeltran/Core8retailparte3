# 📊 Análisis de Ventas Minoristas - Core8 Retail

## 📋 Descripción
Análisis completo de dataset de ventas minoristas con 1000 transacciones. Incluye clasificación de ventas, normalización, agrupación y análisis personalizado con funciones apply.

## 🎯 Funcionalidades
- **Clasificación de Ventas**: Alta (≥$1000), Media (≥$300), Baja (<$300)
- **Clasificación por Edad**: Joven (<30), Adulto (30-49), Adulto Mayor (≥50)
- **Normalización Min-Max**: Escalado de ventas (0-1)
- **Agrupación y Agregación**: Por categoría, mes y edad
- **Análisis con Apply**: Desviaciones, coeficientes de variación, outliers

## 📊 Datos Analizados
- **1000 transacciones** con 9 columnas
- **Categorías**: Beauty, Clothing, Electronics
- **Rango de ventas**: $25 - $2,000
- **Período**: 2023-2024

## 🔍 Métricas Calculadas
- Desviación respecto a media del grupo
- Coeficiente de variación por categoría
- Detección de outliers (32 en Clothing)
- Estadísticas compuestas (media, mediana, asimetría, curtosis)
- Análisis temporal por mes

## 📈 Resultados Destacados
- **Beauty**: Mayor desviación positiva (+$11.48)
- **Clothing**: Mayor número de outliers (32)
- **Electronics**: CV más bajo (1.237)
- **Ventas altas**: 36% en Beauty, 35% en Clothing/Electronics

## 🛠️ Tecnologías
- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Jupyter Notebook

## 📁 Archivos
- `Core8_retail_3.ipynb`: Notebook principal
- `retail_sales_dataset.csv`: Dataset de 1000 transacciones 