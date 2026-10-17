# TCC — qualidade de dados e estabilidade de explicações SHAP

Repositório de desenvolvimento do Trabalho de Conclusão de Curso de Alcimar
Rosal Benvindo Filho, orientado pelo professor Francisco Luciani de Miranda
Vieira.

## Tema

**Efeito de degradações controladas na qualidade dos dados de treinamento
sobre o desempenho preditivo e a estabilidade de explicações SHAP: um estudo
experimental com dados tabulares.**

## Estado do trabalho

- título, problema, objetivo e escopo experimental revisados;
- base Bank Marketing auditada e documentada;
- protocolo experimental aprovado e congelado;
- rotinas auditáveis para valores ausentes, erros nos atributos e ruído nos
  rótulos, com testes automatizados;
- artigo em desenvolvimento no modelo do iCEV;
- piloto de 9 treinamentos concluído e auditado;
- matriz completa de 390 treinamentos concluída e auditada;
- métricas preditivas, SHAP, estabilidade, testes estatísticos, tabelas e
  gráficos gerados;
- H1 e H3 confirmadas; H2 refutada segundo os critérios pré-registrados.

Os resultados completos e sua trilha de auditoria estão em
`experimentos/resultados/completo/`.

## Escopo planejado

- modelos: Regressão Logística, Random Forest e XGBoost;
- degradações: valores ausentes, erros nos atributos e ruído nos rótulos;
- níveis: 5%, 10%, 20% e 30%;
- repetições: dez sementes pareadas;
- desempenho principal: Macro-F1;
- explicações: estabilidade global e local de atribuições SHAP;
- matriz completa executada: 390 treinamentos;
- piloto executado: Random Forest, valores ausentes, níveis de 10% e 30%, três
  sementes e seus baselines, totalizando nove treinamentos.

## Estrutura principal

- `experimentos/PROTOCOLO_EXPERIMENTAL.md`: protocolo científico e estatístico;
- `experimentos/protocolo.json`: parâmetros estruturados do protocolo;
- `experimentos/01_auditoria_bank_marketing.ipynb`: auditoria inicial da base;
- `experimentos/02_resultados_experimento.ipynb`: resultados executados;
- `experimentos/RELATORIO_EXECUCAO_COMPLETA.md`: síntese dos achados;
- `experimentos/src/degradacoes.py`: implementação das degradações;
- `experimentos/tests/`: testes das regras de degradação;
- `modelo-icev-esw-main/2026/artigo-cientifico-tcc/`: fonte LaTeX do artigo.

## Dados e reprodutibilidade

A base Bank Marketing é obtida do UCI Machine Learning Repository, DOI
`10.24432/C5K306`, sob licença CC BY 4.0. Os arquivos brutos não são
versionados. A origem, o hash e a auditoria ficam registrados pelos artefatos
do experimento. Sementes, divisões, máscaras, hiperparâmetros, limiares e
versões das bibliotecas deverão ser preservados em cada execução.

## Testes disponíveis

Na raiz deste repositório:

```powershell
python -m unittest discover -s experimentos/tests -v
```

## Situação do protocolo

O orientador aprovou o protocolo, conforme confirmação do aluno em 17/10/2026.
As emendas estritamente computacionais observadas durante o piloto e a primeira
tentativa da matriz estão registradas na Seção 13 do protocolo.
