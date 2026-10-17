# SHAP Stability under Data Degradation

Projeto experimental e reproduzível sobre o efeito de degradações controladas
nos dados de treinamento, tanto no desempenho preditivo quanto na estabilidade
de explicações SHAP em dados tabulares.

## Objetivo

Avaliar como diferentes problemas de qualidade de dados alteram modelos de
classificação e verificar se as explicações permanecem estáveis quando os dados
de treinamento são progressivamente degradados.

## Desenho experimental

- base: Bank Marketing, do UCI Machine Learning Repository;
- modelos: Regressão Logística, Random Forest e XGBoost;
- degradações: valores ausentes, erros nos atributos e ruído nos rótulos;
- níveis: 5%, 10%, 20% e 30%;
- repetições: dez sementes pareadas;
- desempenho principal: Macro-F1;
- explicações: estabilidade global e local das atribuições SHAP;
- cenário principal sem o atributo `duration`;
- conjunto de teste limpo e fixo.

## Estado atual

- base auditada e documentada;
- protocolo experimental congelado;
- degradações implementadas com testes automatizados;
- piloto de 9 treinamentos concluído;
- matriz completa de 390 treinamentos concluída;
- 120 máscaras de degradação validadas;
- métricas preditivas, SHAP, estabilidade, testes estatísticos, tabelas e
  gráficos gerados;
- H1 e H3 confirmadas e H2 refutada segundo os critérios pré-registrados.

## Estrutura

- `experimentos/PROTOCOLO_EXPERIMENTAL.md`: protocolo científico e estatístico;
- `experimentos/protocolo.json`: parâmetros estruturados;
- `experimentos/01_auditoria_bank_marketing.ipynb`: auditoria inicial da base;
- `experimentos/02_resultados_experimento.ipynb`: análise dos resultados;
- `experimentos/RELATORIO_EXECUCAO_COMPLETA.md`: síntese técnica dos achados;
- `experimentos/src/`: implementação das degradações, treinamento e SHAP;
- `experimentos/tests/`: testes automatizados;
- `experimentos/resultados/completo/`: tabelas, métricas e figuras finais.

## Dados e reprodutibilidade

A base Bank Marketing é obtida do UCI Machine Learning Repository, DOI
`10.24432/C5K306`, sob licença CC BY 4.0. Os dados brutos não são versionados.
A origem, o hash, as sementes, os hiperparâmetros e as versões das bibliotecas
ficam registrados nos artefatos do experimento.

## Testes

Na raiz do projeto:

```powershell
python -m unittest discover -s experimentos/tests -v
```

Para validar os artefatos da execução completa:

```powershell
python experimentos/validate_outputs.py completo
```
