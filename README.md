# Evaluación de Particle Swarm Optimization como técnica de optimización de hiperparámetros en distintos modelos de Machine Learning

## Descripción

Este proyecto implementa y evalúa Particle Swarm Optimization (PSO) como técnica de optimización de hiperparámetros en distintos modelos de Machine Learning. Se compara el desempeño de PSO frente a otros métodos de búsqueda como Random Search y Optimización Bayesiana, utilizando varios datasets y modelos.

## Estructura del repositorio

- `main.py`: Script principal para correr experimentos de optimización de hiperparámetros usando PSO.
- `pso.py`: Implementación del optimizador PSO.
- `models.py`: Wrappers para modelos de ML (kNN, Árbol de Decisión, MLP) con interfaz común para optimización.
- `datasets.py`: Funciones para cargar datasets de prueba (Wine, Digits, sintético multicategoría).
- `experiments.py`: Funciones para correr experimentos y benchmarks comparando PSO, Random Search y Bayes.
- `baselines.py`: Implementaciones de Random Search, Grid Search y BayesOpt (usando scikit-optimize si está disponible).
- `pso_experiments.ipynb`: Notebook con ejemplos, visualizaciones y análisis de resultados.
- `requirements.txt`: Dependencias necesarias para correr el proyecto.
- `figures/`: Carpeta donde se guardan los gráficos generados.

## Requisitos

- Python 3.8+
- Paquetes: numpy, pandas, scipy, matplotlib, scikit-learn
- (Opcional) scikit-optimize (`skopt`) para optimización bayesiana

Instalar dependencias:

```
pip install -r requirements.txt
```

## Cómo correr los experimentos

### 1. Desde el script principal

Ejecutar:

```
python main.py
```

Esto corre PSO para cada combinación de modelo y dataset, mostrando resultados y guardando gráficos de convergencia en la carpeta `figures/`.

### 2. Desde el notebook

Abrir `pso_experiments.ipynb` en Jupyter o VS Code. El notebook permite:
- Probar PSO en distintos modelos y datasets
- Visualizar convergencia y variabilidad
- Comparar PSO vs Random Search vs BayesOpt
- Generar boxplots y tablas resumen

## Modelos y datasets incluidos

- Modelos: kNN, Árbol de Decisión, MLP (red neuronal simple)
- Datasets: Wine, Digits (ambos de scikit-learn), sintético multicategoría

## Resultados esperados

- Gráficos de convergencia de PSO
- Boxplots de accuracies para comparar métodos
- Tablas resumen de mejores hiperparámetros y desempeños