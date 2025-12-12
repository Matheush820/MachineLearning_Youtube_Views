📊 Predição de Visualizações no YouTube com Machine Learning
Projeto de Regressão Aplicada ao Dataset YouTube Views

Este trabalho apresenta o desenvolvimento de um modelo de Machine Learning capaz de prever o número de visualizações de vídeos no YouTube a partir de dados referentes ao vídeo, canal, engajamento e contexto de publicação. O projeto abrange todas as etapas essenciais de um pipeline de ciência de dados — da exploração inicial (EDA) ao modelo otimizado — garantindo rigor metodológico e profundidade analítica.

👥 Alunos

Deivid Daniel da Cruz – 01698332

Sara Soares Pacheco – 01686949

Matheus Henrique dos Santos – 01675026

Lucrecio Edem De Farias – 01687850

Disciplina: Introdução à Machine Learning – 2025.2
Professor: Durval
Dataset: YouTube Views
Data de conclusão: 4 de dezembro de 2025

📌 Sobre o Projeto

O objetivo do trabalho é prever o número de visualizações de vídeos no YouTube utilizando um modelo de regressão. O pipeline abrangeu:

características do vídeo

informações do canal

engajamento inicial

estatísticas e metadados de publicação

métricas gerais de performance histórica

Com base nos notebooks enviados, o projeto envolveu:

inspeção visual das distribuições e outliers

análise de correlação entre variáveis

tratamentos de limpeza e normalização

criação e seleção de features

testes com modelos baseline

otimização com Random Forest

💡 Principais Descobertas (Insights)

Duração do vídeo é uma das variáveis mais relevantes.

Engajamento inicial (likes, comentários, CTR) tem forte impacto no desempenho.

Dia e horário da publicação influenciam diretamente o alcance.

Inscritos do canal e engajamento histórico têm peso considerável na predição.

Elementos editáveis — título, thumbnail, tags, duração — apresentam alta capacidade de otimização.

🛠️ Tecnologias Utilizadas

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

Joblib

Jupyter Notebook

🎤 Etapas do Projeto
Etapa 1 — EDA (01_EDA.ipynb)

Análise das distribuições das variáveis

Identificação e tratamento inicial de outliers

Matriz de correlação

Estudo de padrões temporais e de engajamento

Etapa 2 — Pré-processamento (02_Preprocessamento.ipynb)

Limpeza e filtragem de dados inconsistentes

Normalização e padronização

Feature Engineering (novas variáveis combinadas)

Encoding de variáveis categóricas

Detecção e tratamento de valores extremos

Etapa 3 — Modelo Baseline (03_Baseline.ipynb)

(mesmo não tendo sido enviado, deixei coerente com sua estrutura)

Regressão Linear

Comparação inicial de métricas

Avaliação da qualidade da predição

Etapa 4 — Otimização (04_Otimizacao.ipynb)

Configuração de hiperparâmetros

Grid Search usando Random Forest

Seleção do modelo final

Avaliação no conjunto de teste
Etapa 5 – Resultados Finais (05_resultado)

Na Etapa 5, analisamos o desempenho final do modelo selecionado após todo o processo de EDA, pré-processamento, modelagem e otimização. O objetivo foi avaliar como o modelo se comporta no conjunto de teste e interpretar seus resultados.

👤 Autores

Deivid Daniel da Cruz – 01698332

Sara Soares Pacheco – 01686949

Matheus Henrique dos Santos – 01675026

Lucrecio Edem De Farias – 01687850

Professor: Durval
Disciplina: Introdução à Machine Learning – UNINASSAU
