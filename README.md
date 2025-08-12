# Piattaforma Auto-ML leggera per CSV

Questa app Streamlit consente di:
- Caricare un file CSV
- Selezionare una o più variabili dipendenti (target)
- Eseguire un confronto automatico tra diverse pipeline di ML (preprocessing + selezione feature + modelli + diverse CV)
- Scegliere la combinazione migliore e mostrare le metriche su un test set di hold-out
- Scaricare il modello addestrato

## Requisiti

- Python 3.10+

## Installazione

```bash
pip install -r requirements.txt
```

## Avvio

```bash
streamlit run app.py
```

Carica il tuo CSV dall'interfaccia web, seleziona i target e premi "Esegui Auto-ML".

## Note
- Il sistema rileva automaticamente se si tratta di classificazione o regressione in base al target, ma puoi scegliere la metrica di valutazione.
- La ricerca prova diverse combinazioni di preprocessing (con/ senza selezione feature), vari modelli di `scikit-learn` e differenti numeri di fold per la cross-validation.
- Le metriche mostrate sul test set sono: Accuracy/F1/ROC AUC per classificazione, R2/MAE/RMSE per regressione.
