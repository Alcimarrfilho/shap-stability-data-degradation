# Protocolo experimental — versão congelada e executada

**Status:** protocolo congelado antes da execução. Piloto e matriz completa
executados em 17/10/2026; as emendas computacionais estão registradas na
Seção 13.

**Título confirmado:** *Efeito de degradações controladas na qualidade dos
dados de treinamento sobre o desempenho preditivo e a estabilidade de
explicações SHAP: um estudo experimental com dados tabulares*.

## 1. Perguntas de pesquisa

- **QP1 — desempenho:** como o tipo e o nível de degradação dos dados de
  treinamento alteram o desempenho preditivo dos modelos?
- **QP2 — explicações:** como essas degradações alteram a estabilidade global e
  local das explicações SHAP para as mesmas observações de teste?
- **QP3 — relação:** existem cenários em que o desempenho permanece próximo ao
  baseline, mas as explicações mudam além da variação causada apenas pela troca
  da semente?

## 2. Hipóteses operacionais, definidas antes dos experimentos

- **H1:** no nível de 30%, o ruído nos rótulos reduzirá o Macro-F1 médio em pelo
  menos 0,05 ponto absoluto em relação ao baseline limpo em, no mínimo, dois
  dos três modelos.
- **H2:** haverá ao menos um cenário nos níveis de 20% ou 30% em que a perda
  absoluta de Macro-F1 seja de no máximo 0,02, mas a correlação de Spearman
  entre o ranking SHAP global do baseline e o ranking do cenário degradado seja
  menor ou igual a 0,80.
- **H3:** no nível de 30%, ao menos dois tipos de degradação produzirão, em pelo
  menos um modelo, distância global entre vetores SHAP superior ao percentil 95
  da distância observada entre execuções limpas com sementes diferentes.

As hipóteses poderão ser confirmadas ou refutadas. Os limites numéricos não
serão alterados depois que o piloto for iniciado.

## 3. Base, tarefa e prevenção de vazamento

- Base: **Bank Marketing**, UCI Machine Learning Repository, DOI
  `10.24432/C5K306`, licença CC BY 4.0.
- Arquivo preservado: `bank-full.csv`, com 45.211 registros, 16 atributos
  preditivos e o alvo binário `y`.
- Tarefa: prever se o cliente contratará um depósito a prazo.
- O atributo `duration` será excluído do cenário principal porque só é conhecido
  depois da ligação. Uma análise com esse atributo poderá aparecer apenas em
  apêndice e será identificada como cenário pós-contato, se houver tempo.
- O teste será limpo, fixo e nunca participará de ajuste, imputação, codificação,
  escolha de limiar ou seleção de hiperparâmetros.
- A categoria textual `unknown` será preservada como categoria legítima. Em
  `poutcome`, ela representa ausência de contato anterior e não será convertida
  em valor ausente.

## 4. Particionamento, sementes e pareamento

1. Será criado, uma única vez, um conjunto de teste estratificado de 20%, com semente
   `20260929`. As mesmas observações serão usadas em todas as execuções.
2. Os 80% restantes formarão o conjunto de desenvolvimento. Para cada semente,
   ele será dividido de forma estratificada em treino e validação na proporção
   75/25, resultando em 60%/20%/20% do total.
3. As dez sementes da matriz completa serão:
   `11, 29, 47, 71, 101, 131, 173, 211, 257, 307`.
4. Dentro de cada semente, a mesma divisão e a mesma máscara de degradação serão
   reutilizadas nos três modelos. Isso permite comparações pareadas.
5. Uma amostra estratificada fixa de 500 observações do teste será usada nas
   explicações SHAP. As explicações locais sempre compararão o mesmo cliente.

## 5. Pré-processamento comum

- Colunas numéricas: imputação pela mediana, ajustada exclusivamente no treino
  de cada cenário.
- Colunas categóricas: imputação pela moda exclusivamente para `NaN` sintético e
  codificação one-hot. `unknown` continuará sendo uma categoria.
- As categorias aceitas pelo codificador serão obtidas no treino limpo da
  semente de calibração e depois congeladas, evitando espaços de atributos
  diferentes entre os cenários.
- A Regressão Logística receberá padronização das colunas numéricas. Random
  Forest e XGBoost receberão a mesma imputação e codificação, mas sem
  padronização. Nenhum modelo usará tratamento nativo de ausências.
- Todo ajuste será encapsulado em pipeline para impedir vazamento entre treino,
  validação e teste.

## 6. Degradações controladas

Serão avaliados os níveis de 5%, 10%, 20% e 30%. Para valores ausentes e erros
nos atributos, o nível corresponde à proporção de células preditoras elegíveis
do treino. Para ruído nos rótulos, corresponde à proporção de rótulos dentro de
cada classe. Como as unidades diferem, a inferência principal será feita dentro
de cada tipo de degradação; comparações entre tipos serão apresentadas com essa
limitação explícita.

| Degradação | Regra exata | Controles registrados |
|---|---|---|
| Valores ausentes | Sortear, sem reposição, exatamente `floor(nível × células elegíveis)` pares linha-coluna do treino e substituir os valores por `NaN`. São elegíveis os 15 preditores do cenário principal, exceto células cujo valor original seja `unknown`. | Semente, número total e contagem por atributo; nenhuma alteração em validação, teste, alvo ou `duration`. |
| Erros nos atributos | Sortear o mesmo número de pares linha-coluna. Cada valor será substituído por um valor diferente proveniente de outra linha do mesmo atributo, sorteado no treino. A troca preserva tipo, domínio e distribuição marginal, mas rompe a correspondência correta entre cliente e atributo. | Semente, doador, receptor, atributo, valor anterior e valor substituto; exigir valor diferente. |
| Ruído nos rótulos | Em cada classe do treino, sortear sem reposição `floor(nível × n_classe)` registros e inverter `yes ↔ no`. | Contagens `yes` e `no` antes e depois, quantidade alterada em cada classe e índices afetados. |

As máscaras serão geradas uma vez por semente, tipo e nível e persistidas para
auditoria. As três degradações serão estudadas separadamente; combinações entre
elas ficarão fora do escopo principal.

## 7. Modelos, ajuste e limiar de decisão

1. Regressão Logística, como baseline linear.
2. Random Forest, como conjunto de árvores por bagging.
3. XGBoost, como conjunto de árvores por boosting.

Os hiperparâmetros serão selecionados apenas com os dados limpos da semente de
calibração `20260929`, por validação cruzada estratificada de cinco dobras e
Macro-F1. O espaço de busca e o resultado serão registrados. Depois da escolha,
os hiperparâmetros serão congelados para todas as sementes, degradações e
níveis. Não haverá novo ajuste para favorecer cenários degradados.

O limiar de classificação também será definido somente no conjunto de validação
limpo da semente de calibração. Será escolhida, na grade de 0,05 a 0,95 em passos
de 0,01, a opção com maior Macro-F1; empates serão resolvidos pelo limiar mais
próximo de 0,50. O limiar de cada modelo ficará congelado em todo o experimento.

## 8. Métricas

### Desempenho preditivo

- principal: Macro-F1;
- complementares: PR-AUC, acurácia balanceada e ROC-AUC;
- diagnóstico: matriz de confusão e contagens por classe.

### Estabilidade das explicações SHAP

- Explicadores: `LinearExplainer` para Regressão Logística e `TreeExplainer`
  para Random Forest e XGBoost.
- As contribuições das colunas one-hot serão somadas por atributo original antes
  das comparações, resultando nos mesmos 15 atributos do cenário principal.
- Estabilidade global: correlação de Spearman dos rankings; Jaccard dos cinco
  atributos mais importantes; distância L1 normalizada entre vetores de média
  do valor absoluto de SHAP.
- Estabilidade local: similaridade do cosseno e concordância de sinal entre os
  vetores SHAP das mesmas 500 observações.
- Cada cenário degradado será comparado ao baseline do mesmo modelo e semente.
- A régua de variação natural será formada pelas 45 comparações par a par entre
  as dez execuções limpas de cada modelo. Os percentis 5 e 95 dessa distribuição
  serão registrados antes da interpretação dos cenários degradados.

## 9. Análise estatística

- Para cada modelo, degradação e métrica, aplicar o teste de Friedman aos cinco
  níveis `0, 5, 10, 20, 30` usando as sementes como blocos pareados.
- Havendo diferença global a `α = 0,05`, comparar cada nível ao baseline com o
  teste exato de Wilcoxon pareado e correção de Holm.
- Reportar diferença média e mediana, intervalo bootstrap de 95% e tamanho de
  efeito por correlação bisserial de postos; não apresentar apenas valor-p.
- Avaliar de forma exploratória a associação entre perda de Macro-F1 e
  instabilidade SHAP por correlação de Spearman, sem inferir causalidade.

## 10. Matriz completa e piloto autorizado após revisão

- Matriz completa: `3 modelos × [1 baseline + (3 degradações × 4 níveis)] ×
  10 sementes = 390 treinamentos`.
- Piloto reduzido: **Random Forest**, **valores ausentes**, níveis **10% e 30%**,
  sementes **11, 29 e 47**, mais o baseline correspondente: `1 modelo × 3
  cenários × 3 sementes = 9 treinamentos`.
- Objetivo do piloto: validar carregamento, ausência de vazamento, geração e
  persistência das máscaras, funcionamento do pipeline, cálculo SHAP, formato
  dos resultados, tempo e consumo de memória. O piloto não será usado para
  alterar hipóteses com base em resultados favoráveis.

## 11. Ameaças à validade registradas antecipadamente

- uma única base e um único domínio limitam a validade externa;
- degradações sintéticas não representam todos os mecanismos reais de erro;
- os percentuais de células e de rótulos não são unidades diretamente
  equivalentes;
- desbalanceamento pode tornar algumas métricas instáveis;
- correlação entre atributos e agregação de colunas one-hot podem afetar SHAP;
- dez sementes oferecem estimativas limitadas de incerteza;
- reutilizar o mesmo teste em muitos cenários exige correção de comparações e
  impede qualquer ajuste orientado pelos resultados do teste.

## 12. Reprodutibilidade e critério de congelamento

O repositório registrará código, arquivo original e respectivo hash, regras de
degradação, máscaras, divisões, sementes, hiperparâmetros, limiares, versões do
Python e das bibliotecas, ambiente de execução e tabelas de resultados. Depois
da revisão final, esta versão receberá um identificador de protocolo e será
congelada antes do piloto. Qualquer desvio posterior será documentado com
data, motivo e impacto esperado.

## 13. Emenda computacional posterior ao congelamento — 17/10/2026

O primeiro treinamento do piloto mostrou que o cálculo exato do TreeSHAP para
500 observações da Random Forest exigiu 216 segundos, o que tornaria a matriz
completa desproporcionalmente longa. Sem alterar base, partições, modelos,
hiperparâmetros, sementes, amostra de 500 observações, agregação por atributo ou
métricas, o `TreeExplainer` será executado com a opção determinística
`approximate=True` e `check_additivity=False` nos modelos de árvore.

Antes de retomar o piloto, a aproximação será comparada ao TreeSHAP exato nas
primeiras 50 observações da amostra fixa do baseline Random Forest, semente 11.
A emenda será aceita somente se a correlação de Spearman entre importâncias
globais for pelo menos 0,95 e a similaridade local de cosseno média for pelo
menos 0,90. O resultado dessa validação será persistido. A Regressão Logística
continuará usando `LinearExplainer`, sem aproximação.

Durante a primeira tentativa da matriz completa, o SHAP 0.49.1 não conseguiu
interpretar o intercepto vetorial introduzido no XGBoost 3.1. A execução parou
após dois artefatos, antes do primeiro resultado XGBoost. O SHAP foi atualizado
para 0.54.1, versão compatível com esse formato; os resultados já calculados
serão refeitos com `--forcar` para manter um único ambiente em toda a matriz.

Na reinicialização, o SHAP 0.54.1 informou que o `LinearExplainer` reduziria
automaticamente o conjunto de referência de 200 para 100 observações. Para não
delegar uma subamostragem oculta à biblioteca, o `masker` foi configurado
explicitamente com `max_samples=200`. A matriz foi reiniciada antes da análise.

