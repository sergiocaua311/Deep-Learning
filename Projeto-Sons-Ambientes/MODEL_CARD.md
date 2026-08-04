# Model card do projeto ESC-50

## Uso pretendido

Comparar encoders de áudio e classificar clipes nas 50 categorias fechadas do
ESC-50. O projeto é educacional e experimental; não é um detector universal de
eventos acústicos.

## Avaliação

A triagem usa folds 1–3 para treino, fold 4 para validação e fold 5 como teste
cego. Os dois melhores modelos seguem para os cinco folds oficiais, com
hiperparâmetros congelados. O vencedor pode depois ser treinado nos 2.000 clipes
para inferência, sem atribuir a esse último checkpoint uma métrica independente.

## Métricas

F1 macro é a métrica principal. Também são registrados acurácia, top-5, log
loss, erro de calibração, latência, tamanho e queda sob perturbações.

## Limitações

- O conjunto é pequeno, balanceado e não representa toda a variedade acústica real.
- O modelo sempre escolhe uma das 50 classes, mesmo para sons fora do domínio.
- Latência em MPS, CUDA e CPU não é diretamente comparável.
- O resultado histórico de 92,75% do AST selecionou checkpoint no fold 5 e é apenas referência.
