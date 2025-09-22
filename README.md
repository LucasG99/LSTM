# LSTM — detecção de fraude no `creditcard.csv`

Eu queria ver se uma LSTM consegue capturar padrões no tempo para separar fraudes de transações legítimas. Fiz o básico com cuidado: entendi o dado, respeitei a ordem temporal, evitei vazamento e avaliei com métricas que fazem sentido para base desbalanceada.

## O que eu entrego (alinhado ao barema)
- Carrego o `creditcard.csv` e faço EDA (distribuições, correlações, outliers) com leitura curta do que cada gráfico me disse.
- Preparo os dados **sem vazamento**: ordeno por `Time`, split temporal 70/15/15 (treino/val/teste), imputo e padronizo **fitando só no treino** e aplicando em val/teste, e crio **sequências temporais** para a LSTM.
- Defino a **arquitetura** (LSTM 64→32 com `Dropout`, `Adam`, `binary_crossentropy`, `class_weight`) e justifico as escolhas.
- **Treino/valido/testo** e reporto **Accuracy, Precision, Recall, F1, ROC-AUC** (e PR-AUC/AP). Ploto as **curvas de aprendizado** e comento over/underfitting olhando o gap de AUC (treino vs validação) e a tendência da `val_loss`.
- Analiso resultados, comparo pontos de operação (threshold 0,5 vs limiar ajustado pela validação) e encerro com uma conclusão direta.

## Como rodar
1. Coloque `creditcard.csv` na raiz do repositório.
2. Abra o notebook `LTSM_2.ipynb` e execute as células na ordem.
3. (Opcional) Use um ambiente virtual e instale as dependências abaixo.

**Dependências mínimas**: numpy, pandas, matplotlib, seaborn, scikit-learn, tensorflow>=2.10

## O que o notebook faz 
- **EDA**: distribuição da classe, `Amount`, correlações/KDEs; comento o que cada gráfico mostra e por que importa.
- **Preparação sem vazamento**: split temporal, imputação e `StandardScaler` ajustados só no treino; criação das sequências (janela de 30, stride 5).
- **Modelo**: LSTM (64→32) + `Dropout`; `class_weight` para a minoria; `EarlyStopping` em `val_auc` com restauração dos melhores pesos.
- **Avaliação**: Accuracy, Precision, Recall, F1, ROC-AUC e PR-AUC; curvas ROC/PR; matrizes de confusão.
- **Threshold tuning**: ajusto o limiar pela validação (maximizei F1 nesta entrega) e comparo com o 0,5 em teste numa tabela.

## Conclusão
O pipeline está correto (ordem temporal e nada de vazamento). A LSTM aprendeu um sinal útil e o ponto de operação muda com o limiar: ao **ajustar o threshold pela validação** eu consegui melhorar o objetivo escolhido (nesta entrega, F1), aceitando o trade-off esperado entre precisão e recall. Se a prioridade fosse **pegar mais fraudes**, eu ajustaria o limiar mirando **recall** e reforçaria a regularização (ou testaria `class_weight` mais forte/focal loss). A leitura das **curvas de aprendizado** e de **val vs teste** fecha a análise de over/underfitting sem achismo.
