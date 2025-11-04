
---

## 🧰 Bibliotecas Utilizadas

| Biblioteca | Função Principal |
|-------------|------------------|
| **pandas** | Manipulação e análise de dados tabulares |
| **numpy** | Cálculos numéricos e estatísticos |
| **matplotlib** | Criação de gráficos e visualizações básicas |
| **seaborn** | Visualizações estatísticas e heatmaps de correlação |
| **scipy** | Testes estatísticos e medidas de distribuição |

---

## 📖 Estrutura do Notebook

O notebook `01_EDA.ipynb` está dividido nas seguintes seções:

1. **Importação de Bibliotecas**  
   - Carregamento e configuração inicial das bibliotecas usadas.

2. **Carregamento dos Dados**  
   - Leitura do arquivo `youtube_views.csv`.  
   - Verificação de dimensões, tipos e primeiras amostras do dataset.

3. **Visão Geral do Dataset**  
   - Separação de variáveis numéricas e categóricas.  
   - Identificação da variável alvo (`final_grade` ou substituta, conforme dataset).

4. **Análise de Valores Faltantes**  
   - Quantificação e visualização dos `NaN`.  
   - Investigação de padrões e possíveis causas.

5. **Análise da Variável Alvo**  
   - Estatísticas descritivas, histogramas e boxplots.  
   - Teste de normalidade (Shapiro-Wilk).

6. **Análise Univariada — Variáveis Numéricas**  
   - Distribuições, medidas centrais e detecção de outliers (IQR).

7. **Análise Univariada — Variáveis Categóricas**  
   - Frequência e proporção das categorias.  
   - Identificação de desbalanceamentos e inconsistências.

8. **Análise de Correlações**  
   - Matriz de correlação entre variáveis numéricas.  
   - Heatmap para visualização e interpretação.

9. **Análise Bivariada (Features vs Target)**  
   - Relação entre variáveis categóricas e a variável alvo.  
   - Boxplots e comparação de médias.

10. **Identificação de Outliers**  
    - Detecção e contagem de outliers em todas as variáveis numéricas.  
    - Discussão sobre legitimidade dos valores.

11. **Conclusões e Descobertas Principais**  
    - Resumo executivo com:
      - Tamanho e qualidade geral do dataset  
      - Problemas detectados (faltantes, outliers, inconsistências)  
      - Principais insights sobre o comportamento das variáveis  
      - Features mais relevantes  
      - Próximos passos recomendados

---

## 📈 Critérios de Avaliação

| Critério | Peso | Descrição |
|-----------|------|------------|
| **Completude** | 30% | Todas as 11 seções e análises obrigatórias estão presentes |
| **Visualizações** | 20% | Gráficos claros, com títulos, labels e legendas |
| **Documentação** | 25% | Markdown explicativo, interpretações e conclusões bem redigidas |
| **Qualidade Técnica** | 15% | Notebook executa do início ao fim sem erros |
| **Insights** | 10% | Descobertas e interpretações relevantes |

---

## 🚫 Erros Comuns a Evitar

- ❌ Tratar dados (preencher, remover, codificar) nesta etapa  
- ❌ Gerar gráficos sem título ou legenda  
- ❌ Deixar código sem explicação ou markdown  
- ❌ Análises superficiais sem interpretação

---

## 💡 Dicas de Sucesso

- Use **headers markdown** e um **índice** no início do notebook  
- Comente o código quando for mais complexo  
- Faça **gráficos limpos e legíveis**  
- Explore **padrões interessantes** nos dados  
- Finalize com um resumo executivo bem redigido

