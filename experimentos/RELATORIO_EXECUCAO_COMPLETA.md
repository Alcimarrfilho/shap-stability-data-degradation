# Relatório da execução experimental completa

## Situação

O protocolo aprovado foi implementado e executado. A auditoria automática
confirmou 390 treinamentos, 120 máscaras de degradação e nenhum erro estrutural.
Foram usados dez particionamentos, três modelos, um baseline e doze cenários
degradados por modelo e semente. O teste limpo e a amostra SHAP de 500 registros
permaneceram fixos.

## Calibração congelada

| Modelo | Macro-F1 médio no baseline | Limiar congelado |
|---|---:|---:|
| Regressão Logística | 0,7019 | 0,21 |
| Random Forest | 0,7091 | 0,55 |
| XGBoost | 0,7054 | 0,27 |

Os hiperparâmetros foram escolhidos uma única vez nos dados limpos por validação
cruzada estratificada de cinco dobras. O arquivo
`modelos/configuracao_congelada.json` contém os parâmetros completos.

## Resultado principal no nível de 30%

Os valores abaixo são médias das dez sementes. `Δ Macro-F1` é a diferença
pareada em relação ao baseline do mesmo modelo e semente. Valores menores de
Spearman e cosseno indicam explicações menos estáveis; valores maiores de L1
indicam maior distância entre os vetores globais de importância.

| Modelo | Degradação | Δ Macro-F1 | Spearman SHAP global | L1 global | Cosseno SHAP local |
|---|---|---:|---:|---:|---:|
| Regressão Logística | Erros em atributos | -0,0241 | 0,8464 | 0,4243 | 0,7698 |
| Regressão Logística | Ruído em rótulos | -0,5971 | 0,7721 | 0,4897 | 0,8097 |
| Regressão Logística | Valores ausentes | -0,0117 | 0,9382 | 0,1420 | 0,9671 |
| Random Forest | Erros em atributos | -0,0388 | 0,7989 | 0,3538 | 0,7582 |
| Random Forest | Ruído em rótulos | -0,0511 | 0,7329 | 0,3507 | 0,5968 |
| Random Forest | Valores ausentes | -0,0067 | 0,9561 | 0,1259 | 0,9175 |
| XGBoost | Erros em atributos | -0,0236 | 0,7979 | 0,2775 | 0,5840 |
| XGBoost | Ruído em rótulos | -0,3803 | 0,7168 | 0,3814 | 0,3681 |
| XGBoost | Valores ausentes | -0,0119 | 0,9314 | 0,1271 | 0,7715 |

No nível de 30%, todas as nove comparações de Macro-F1 foram inferiores ao
baseline em todas as sementes: Wilcoxon exato com correção de Holm `p = 0,0078`
e correlação bisserial de postos `-1,0`. A magnitude, porém, variou muito. O
ruído nos rótulos foi especialmente destrutivo para Regressão Logística e
XGBoost; valores ausentes produziram as menores perdas de Macro-F1.

## Hipóteses pré-registradas

- **H1 confirmada.** Com 30% de ruído nos rótulos, a redução média de Macro-F1
  foi de 0,5971 na Regressão Logística, 0,0511 na Random Forest e 0,3803 no
  XGBoost. O limite era redução mínima de 0,05 em pelo menos dois modelos.
- **H2 refutada.** Nenhuma combinação média de modelo, degradação e nível de
  20% ou 30% apresentou simultaneamente perda absoluta de Macro-F1 de até 0,02
  e Spearman SHAP global de até 0,80.
- **H3 confirmada.** No nível de 30%, os três tipos de degradação ultrapassaram,
  em todos os modelos, o percentil 95 da distância L1 observada entre baselines
  treinados com sementes diferentes.

H2 refutada não significa que desempenho e explicações sempre caminham juntos.
Por exemplo, 30% de valores ausentes causaram pequena perda média de Macro-F1,
mas sua distância L1 global já ultrapassou a variação natural nos três modelos.
O que não ocorreu foi o limiar mais severo de Spearman `≤ 0,80` definido em H2.

## Controle de qualidade e limitações

- O atributo `duration` foi excluído para evitar vazamento temporal.
- As degradações atingiram somente o treino; validação e teste permaneceram
  limpos.
- O SHAP 0.54.1 foi necessário para compatibilidade com o intercepto vetorial do
  XGBoost 3.1.1.
- O TreeSHAP aproximado foi adotado por custo computacional e validado antes da
  matriz contra o cálculo exato em 50 observações: Spearman global 0,9536,
  cosseno local médio 0,9766 e Jaccard top-5 igual a 1,0.
- Uma única base limita a generalização; as degradações são sintéticas e suas
  unidades não são diretamente equivalentes; dez sementes limitam a precisão
  da incerteza.

## Artefatos para conferência

- `02_resultados_experimento.ipynb`: caderno executado, com tabelas e figuras;
- `resultados/completo/tabela_resultados_principais.csv`: tabela consolidada;
- `resultados/completo/testes_estatisticos.csv`: Friedman, Wilcoxon-Holm,
  intervalos e tamanho de efeito;
- `resultados/completo/avaliacao_hipoteses.json`: decisão automática de H1–H3;
- `resultados/completo/figuras/`: cinco figuras finais;
- `resultados/completo/validacao_artefatos.json`: comprovante da auditoria;
- `resultados/manifesto_execucao.json`: versões e hash SHA-256 da base;
- `resultados/validacao_shap_aproximado.json`: validação da aproximação SHAP.
