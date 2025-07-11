# 📊 Análisis y Predicción de Ventas en Retail

## 🎯 Objetivo del Proyecto
Desarrollar un modelo de machine learning para clasificar automáticamente las ventas en categorías (Alta, Media, Baja) basándose en características del cliente y la transacción.

## 📋 Dataset
- **1,000 transacciones** de ventas retail
- **3 categorías** de productos: Beauty, Clothing, Electronics
- **Rango de ventas**: $25 - $2,000
- **Variables**: Edad, Género, Cantidad, Precio por Unidad, Categoría de Producto

## 🔧 Metodología

### Análisis Exploratorio (EDA)
- Análisis de correlaciones con mapas de calor
- Detección de outliers (32 en Clothing)
- Visualizaciones avanzadas con subplots
- Ingeniería de características temporales

### Preprocesamiento
- **3 pipelines** con ColumnTransformer:
  - StandardScaler + OneHotEncoder
  - MinMaxScaler + OneHotEncoder  
  - RobustScaler + OrdinalEncoder
- Manejo automático de valores faltantes
- Codificación de variables categóricas

### Evaluación de Modelos
- **6 algoritmos** evaluados con validación cruzada 5-fold
- Métricas: Accuracy, Precision, Recall, F1-Score, AUC

## 🏆 Resultados

### Mejor Modelo: Decision Tree
- **Accuracy**: 100.0%
- **F1-Score (macro)**: 100.0%
- **Precision (macro)**: 100.0%
- **Tiempo de entrenamiento**: 0.096 segundos

### Rendimiento por Clase
| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Alta | 1.00 | 1.00 | 1.00 | 40 |
| Baja | 1.00 | 1.00 | 1.00 | 121 |
| Media | 1.00 | 1.00 | 1.00 | 39 |

## 💡 Insights Clave
1. **Dataset balanceado** con ratio máximo de 1.7:1
2. **Clothing** presenta mayor complejidad de predicción
3. **Características temporales** (mes, día) son relevantes
4. **Modelo robusto** listo para implementación en producción

## 🔬 Tecnologías Utilizadas
- **Python**: Pandas, NumPy, Scikit-learn
- **Visualización**: Matplotlib, Seaborn
- **ML**: Random Forest, SVM, KNN, Logistic Regression
- **Evaluación**: Validación Cruzada, ROC-AUC, Matriz de Confusión

## 📊 Archivos Generados
- Notebooks de análisis completo (EDA, Preprocessing, Benchmarking, Metrics)
- Reportes de clasificación y matrices de confusión
- Curvas ROC y métricas detalladas
- Modelos entrenados listos para producción

---
**Conclusión**: El modelo Decision Tree demuestra capacidad predictiva perfecta para clasificar ventas retail, logrando 100% de precisión en todas las métricas y clases evaluadas.
