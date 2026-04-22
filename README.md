# A IO Project - 100B MoE Model

Este repositorio contiene la arquitectura oficial de **A IO**, un modelo de lenguaje masivo basado en **Mixture of Experts (MoE)**.

## Especificaciones Técnicas
- **Arquitectura:** Transformer MoE.
- **Parámetros Totales:** 100 Billones (100B).
- **Configuración de Expertos:** 16 expertos totales, con 2 activos por cada proceso.
- **Dataset de Pre-entrenamiento:** Wikimedia Corpus (vía Streaming).

## Objetivo
El proyecto A IO busca utilizar la capacidad de computación masiva distribuida para entrenar un modelo desde cero, optimizado para ser eficiente y potente.

## Estructura de Archivos
- `model.py`: Definición de la arquitectura neuronal.
- `pretrain.py`: Lógica para el entrenamiento masivo.
- `config.yaml`: Ajustes y dimensiones del modelo.
- `tokenizer.py`: Sistema de procesamiento de texto.
