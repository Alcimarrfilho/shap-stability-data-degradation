# Ambiente de execução

## Ambiente executado em 17 de outubro de 2026

- Ambiente virtual local: `.venv`.
- Python: 3.13.15.
- NumPy 2.3.4; pandas 2.3.3; scikit-learn 1.7.2; XGBoost 3.1.1;
  SHAP 0.54.1; SciPy 1.16.3; Matplotlib 3.10.7.
- As dependências diretas estão congeladas em `requirements.txt`; as versões
  efetivamente carregadas e o hash da base constam em
  `resultados/manifesto_execucao.json`.

O protocolo foi congelado antes da execução. Os 12 testes automatizados, o
piloto de 9 execuções e a matriz de 390 execuções foram concluídos. A auditoria
final confirmou todos os artefatos esperados sem erros.
