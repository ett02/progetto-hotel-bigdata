# Analisi Backend e Utilizzo Spark nelle Query

Questo documento descrive in dettaglio come Apache Spark viene utilizzato in `backend.py` per ogni funzione di analisi ("query"). Sebbene il codice utilizzi le API DataFrame di alto livello (PySpark SQL), i concetti sottostanti di RDD, Map e Reduce sono fondamentali per l'esecuzione distribuita.

In PySpark DataFrame API:
- **Creazione RDD**: Avviene al caricamento (es. `spark.read.csv`) o come risultato di una trasformazione. Ogni nuovo DataFrame rappresenta implicitamente un nuovo RDD nella lineage di esecuzione.
- **Cache/Persist**: Fondamentale in Spark per ottimizzare algoritmi iterativi (ML) o riutilizzi multipli dello stesso DataFrame.
- **Map**: Trasformazioni "row-wise" (per riga) che non richiedono shuffle (es. `select`, `filter`, `withColumn`, `udf`).
- **Reduce**: Aggregazioni che richiedono shuffle per raggruppare dati (es. `groupBy().agg()`, `distinct`, `join`).

---

## 1. `query_nazionalita_critiche`
**Descrizione**: Analizza i voti medi per nazionalità, filtrando quelle con poche recensioni.

*   **Creazione RDD/DataFrame**: Input `df_hotel`.
*   **Creazione Nuovo RDD**:
    1.  `df_n`: creato tramite trasformazione `withColumn` (pulizia stringa).
    2.  `result`: creato dall'aggregazione e dai successivi raffinamenti.
*   **Map (Pre-Aggregation)**:
    -   `trim(col("Reviewer_Nationality"))`: pulizia stringa su ogni riga.
*   **Reduce (Shuffle)**:
    -   `groupBy("nationality_clean").agg(...)`: raggruppa tutte le righe con la stessa nazionalità e calcola media, conteggio, dev. std, min e max.
*   **Map (Post-Aggregation)**:
    -   `filter(col("num_recensioni") > min_reviews)`: filtra i risultati aggregati (operazione row-wise sul dataset ridotto).
    -   `withColumn(...)`: arrotondamenti finali.

---

## 2. `query_impatto_costruzioni`
**Descrizione**: Analizza l'impatto di lavori in corso (keyword construction/renovation) sui voti usando regex.

*   **Creazione RDD/DataFrame**: Input `df_hotel`.
*   **Creazione Nuovo RDD**:
    1.  `df_c`: DataFrame arricchito con colonne booleane per presenza keyword.
    2.  `stats_df`, `keywords_df`, `samples_df`: tre diversi DataFrame derivati da `df_c`.
*   **Map (Pre-Aggregation)**:
    -   `rlike(pattern)`: map booleana per identificare righe con lavori.
    -   `regexp_extract(...)`: estrazione della specifica parola trovata.
*   **Reduce (Shuffle)**:
    -   `stats_df`: `groupBy("has_construction").agg(...)` riduce l'intero dataset a 2 righe (True/False).
    -   `keywords_df`: `groupBy("kw_main").agg(count)` riduce per contare frequenze delle parole.
*   **Map (Post-Aggregation)**:
    -   Su `stats_df`: Calcolo intervallo di confidenza (`ci95`) e Standard Error (`se`) usando colonne aggregate.
    -   Su `keywords_df`: Ordinamento (`orderBy`).
    -   Su `samples_df`: Filtro e ordinamento casuale (`rand()`).

---

## 3. `query_coppie_vs_famiglie`
**Descrizione**: Confronta voti medi tra diverse tipologie di viaggiatori (Coppie, Famiglie, ecc.) derivate dai tag.

*   **Creazione RDD/DataFrame**: Input `df_hotel`.
*   **Creazione Nuovo RDD**: `df_viaggi` (classificato) e `agg` (statistiche).
*   **Map (Logica Condizionale)**:
    -   `withColumn("tipo_viaggio", ...)`: Una complessa catena di `when/otherwise` che mappa ogni riga in una categoria basata sul contenuto della stringa `Tags`.
*   **Reduce (Shuffle)**:
    -   `groupBy("tipo_viaggio").agg(...)`: Aggregazione per calcolare media e deviazione standard per categoria.
*   **Map (Post-Aggregation)**:
    -   Calcoli statistici (`se`, `ci95`) sui risultati aggregati.
    -   Filtro (`num_recensioni > 50`) per escludere categorie poco rappresentative.

---

## 4. `query_lunghezza_recensioni`
**Descrizione**: Analizza l'asimmetria del sentiment basandosi sulla lunghezza del testo.

*   **Creazione RDD/DataFrame**: Input `df_hotel`.
*   **Creazione Nuovo RDD**: `df_bucket` (categorizzato per score) e `result` (finale).
*   **Map (Binning)**:
    -   `withColumn("bucket_id", ...)`: Discretizzazione del punteggio continuo `Reviewer_Score` in bucket categorici.
*   **Reduce (Shuffle)**:
    -   `groupBy("bucket_id").agg(...)`: Calcola la lunghezza media del testo per bucket.
    -   Nota: L'espressione `count(when(col(...) > 0, 1))` all'interno dell'agg è un pattern comune in Spark per fare "Filter-then-Count" durante la fase di reduce senza fare shuffle aggiuntivi.
*   **Map (Post-Aggregation)**:
    -   Calcolo differenze (`delta`) e rapporti (`ratio`) tra le colonne aggregate.

---

## 5. `query_affidabilita_voto`
**Descrizione**: Identifica hotel con alta varianza nei voti (CV Score).

*   **Creazione RDD/DataFrame**: Input `df_hotel`.
*   **Reduce (Shuffle)**:
    -   `groupBy("Hotel_Name").agg(...)`: Riduce tutte le recensioni di un hotel a una singola riga di statistiche.
*   **Map (Post-Aggregation)**:
    -   `filter(num_reviews >= 100)`: Rimuove hotel non statisticamente significativi.
    -   Calcolo `cv_score` (Coefficiente di Variazione): Operazione aritmetica (`stddev / mean`) su ogni riga del dataset ridotto.

---

## 6. `query_hotel_rischiosi`
**Descrizione**: Identifica hotel con buona media ma alta probabilità di esperienze disastrose (Coda sinistra della distribuzione).

*   **Creazione RDD/DataFrame**: Input `df_hotel`.
*   **Reduce (Shuffle)**:
    -   `groupBy("Hotel_Name").agg(...)`: Calcola statistiche. Include `percentile_approx`, un algoritmo di approssimazione per calcolare percentili in modo distribuito ed efficiente.
*   **Map (Post-Aggregation)**:
    -   Calcolo `disaster_pct` e `risk_index` (formule matematiche sulle colonne aggregate).
    -   Filtro complesso (`avg >= 8.0` AND `disaster_pct >= 5.0`).

---

## 7. `query_expectation_gap`
**Descrizione**: Analizza il gap tra aspettativa (punteggio medio hotel) e realtà (voto singola recensione).

*   **Creazione RDD/DataFrame**: Input `df_hotel`.
*   **Map (Row-wise)**:
    -   `col("Reviewer_Score") - col("Average_Score")`: Calcolo del gap per ogni singola recensione.
    -   Binning del prestigio hotel (assegnazione bucket).
*   **Reduce (Shuffle)**:
    -   `groupBy("bucket_id").agg(...)`: Aggrega per fascia di prestigio, calcolando la media dei gap e la % di delusioni.

---

## 8. Funzioni ML Avanzate

### `allena_sentiment_hotel` (Logistic Regression)
*   **Utilizzo Spark**: Pipeline ML completa.
*   **Map (Feature Engineering)**:
    -   `Tokenizer`, `StopWordsRemover`, `VectorAssembler`: Trasformazioni che convertono testo grezzo in vettori numerici riga per riga.
*   **Reduce (Training)**:
    -   `CountVectorizer.fit`: Richiede una passata completa sui dati per costruire il vocabolario (Reduce).
    -   `IDF.fit`: Richiede una passata per calcolare la df (document frequency).
    -   `LogisticRegression.fit`: Algoritmo iterativo che usa aggregazioni distribuite (reduce) per calcolare il gradiente e aggiornare i pesi.
*   **Tuning/Cache**:
    -   Il codice esegue esplicitamente `train_data.cache()` per mantenere in memoria i dati di training processati, accelerando le iterazioni multiple della Logistic Regression.

### `esegui_clustering_hotel` (K-Means)
*   **Map (Feature Engineering)**:
    -   Preprocessing iniziale (aggregazione per hotel) -> *Nota: questo step include una reduce preliminare per preparare il dataset "per hotel"*.
    -   `VectorAssembler` e `StandardScaler`: Normalizzazione delle feature.
*   **Reduce (Training)**:
    -   `KMeans.fit`: Algoritmo iterativo (EM-style). Alterna fasi di assegnazione (Map: calcolo distanza dai centroidi) e aggiornamento (Reduce: ricalcolo posizione centroidi).

### `esegui_topic_modeling` (LDA)
*   **Utilizzo Spark**: Topic Modeling su testo non strutturato.
*   **Map (Cleaning)**:
    -   Filtro iniziale (`filter`) per rimuovere recensioni "No Negative".
    -   `RegexTokenizer` e `StopWordsRemover`: pulizia testo.
*   **Reduce (Training)**:
    -   `CountVectorizer`: Costruzione vocabolario.
    -   `LDA.fit` (Latent Dirichlet Allocation): Algoritmo probabilistico complesso e computazionalmente intensivo che richiede molteplici passaggi sui dati.
*   **Cache/Persist**:
    -   Il codice usa `df_neg.persist(StorageLevel.MEMORY_AND_DISK)` perché il DataFrame filtrato viene riutilizzato più volte (una volta per il fit del modello principale, e potenzialmente una seconda volta per lo stability check con seed diverso).
    -   Chiude con `df_neg.unpersist()` per liberare risorse.
