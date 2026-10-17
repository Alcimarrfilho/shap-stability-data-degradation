# Experimentos

Esta pasta reúne a auditoria da Bank Marketing, o protocolo aprovado, o código
reproduzível e os artefatos da execução experimental.

## Estado atual

- A base original foi preservada e auditada.
- `duration` está excluído do cenário principal.
- O protocolo foi validado e congelado antes da execução.
- O piloto de 9 treinamentos e a matriz completa de 390 foram executados.
- A auditoria confirmou 390 execuções, 120 máscaras e zero erros estruturais.
- H1 e H3 foram confirmadas; H2 foi refutada nos critérios pré-registrados.

## Arquivos centrais

- `PROTOCOLO_EXPERIMENTAL.md`: decisões científicas e estatísticas.
- `protocolo.json`: parâmetros legíveis pelo código.
- `01_auditoria_bank_marketing.ipynb`: auditoria reproduzível da base.
- `02_resultados_experimento.ipynb`: leitura executável dos resultados.
- `RELATORIO_EXECUCAO_COMPLETA.md`: síntese técnica e interpretação cautelosa.
- `run_experiment.py`: calibração, piloto e matriz completa.
- `analyze_results.py`: estabilidade, estatística, tabelas e gráficos.
- `src/degradacoes.py`: geração auditável das três degradações.
- `src/experimento.py`: pipeline completo de treinamento e SHAP.
- `tests/`: 12 verificações automatizadas.

## Reprodução

Crie o ambiente com `requirements.txt`, execute os testes e use, nesta ordem,
`run_experiment.py calibrar`, `run_experiment.py piloto`,
`validate_outputs.py piloto`, `run_experiment.py completo`,
`validate_outputs.py completo` e `analyze_results.py completo`. As versões e o
hash da base usados nesta execução estão em `resultados/manifesto_execucao.json`.
