# Roteiro sugerido — vídeo de entrega

## 0:00–0:45 — Introdução

"Meu nome é Filipe Martins Henrique. O tema escolhido é Pima Indians Diabetes. O objetivo desta atividade foi reorganizar o treinamento de uma rede neural em um pipeline observável com MLflow, permitindo rastrear, comparar e reproduzir os experimentos."

## 0:45–2:00 — Estrutura e dados

Mostrar a pasta do projeto e explicar `data.py`, `model.py`, `train.py`, `pipeline.py` e `main.py`.

Explicar que cada run usa a mesma seed e refaz o mesmo split estratificado 70/15/15. O imputer e o scaler são ajustados **somente no conjunto de treino** para evitar vazamento de informação.

Comentar que zeros em `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` e `BMI` são tratados como ausentes e imputados pela mediana calculada apenas no treino.

## 2:00–3:30 — Experimentos

Mostrar no código as três configurações:

- baseline: learning rate 0.01, sem L2;
- lr_menor: learning rate 0.001, sem L2;
- regularizacao_l2: learning rate 0.01, weight decay 1e-5.

Dizer que arquitetura, split, seed, batch size e épocas são iguais para tornar a comparação justa.

## 3:30–5:30 — MLflow

Abrir a interface do MLflow e mostrar:

- três runs;
- parâmetros;
- `train_loss`, `val_loss`, `val_accuracy` e `val_f1` por época;
- `final_val_precision`, `final_val_recall` e `final_val_f1`;
- curvas de loss e métricas de validação;
- modelo salvo como artefato em cada run;
- preprocessing salvo em cada run;
- traces `prepare_data`, `train` e `validate`.

## 5:30–7:00 — Escolha do modelo

Comparar as runs na validação (`n_val = 115`). Explicar que a escolha foi feita pela maior F1 de validação e que a acurácia foi usada como desempate:

- `baseline` (`lr=0.01`, `L2=0`): `val_accuracy = 0.7217`, `val_precision = 0.5870`, `val_recall = 0.6750`, `val_f1 = 0.6279`
- `lr_menor` (`lr=0.001`, `L2=0`): `val_accuracy = 0.7217`, `val_precision = 0.5952`, `val_recall = 0.6250`, `val_f1 = 0.6098`
- `regularizacao_l2` (`lr=0.01`, `L2=1e-5`): `val_accuracy = 0.7826`, `val_precision = 0.6744`, `val_recall = 0.7250`, `val_f1 = 0.6988`

**Fala sugerida:**

"A run escolhida foi **`regularizacao_l2`**, pois obteve F1 de validação de **0.6988 (69,88%)** e acurácia de validação de **0.7826 (78,26%)**, superando tanto a `baseline` (F1 de 0.6279) quanto a `lr_menor` (F1 de 0.6098). Na run escolhida, a precision de validação foi **0.6744 (67,44%)** e o recall foi **0.7250 (72,50%)**."

## 7:00–8:15 — Teste final

Mostrar que somente após a escolha o modelo `regularizacao_l2` foi avaliado no conjunto de teste reservado (`n_test = 116`).

**Fala sugerida:**

"No conjunto de teste, o modelo escolhido (`regularizacao_l2`) obteve acurácia de **0.7328 (73,28%)**, precision de **0.6389 (63,89%)**, recall de **0.5610 (56,10%)** e F1 de **0.5974 (59,74%)**."

Mostrar a matriz de confusão (`TN=62`, `FP=13`, `FN=18`, `TP=23`), o relatório de classificação (`classification_report_test.json`) e o span `final_test` exclusivo da run selecionada.

## 8:15–9:00 — Diagnóstico e conclusão

Comentar o comportamento das curvas `train_loss` e `val_loss`:

- Nas configurações com `learning_rate = 0.01` (`baseline` e `regularizacao_l2`), a `train_loss` caiu para perto de zero (`0.0278` e `0.0394`), enquanto a `val_loss` subiu após as primeiras dezenas de épocas (`1.7854` e `2.0419`), evidenciando sobreajuste (*overfitting*) na calibração das probabilidades ao longo de 100 épocas. Mesmo assim, a penalidade L2 (`weight_decay = 1e-5`) regularizou os pesos o suficiente para melhorar a fronteira de decisão na validação (`val_f1 = 0.6988`).
- Na configuração `lr_menor` (`learning_rate = 0.001`), o aprendizado foi mais conservador e controlou melhor a perda de validação (`val_loss = 0.6096`), mas obteve menor recall (`0.6250`) e F1 (`0.6098`) ao fim de 100 épocas.

Finalizar:

"O MLflow permitiu associar cada resultado à configuração que o produziu, registrar parâmetros, métricas, artefatos e traces (`prepare_data`, `train`, `validate` e `final_test`), e reproduzir o pipeline sem alterar manualmente o código a cada experimento."

