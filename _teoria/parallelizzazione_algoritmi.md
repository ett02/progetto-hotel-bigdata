# Analisi Approfondita della Parallelizzazione in PySpark

Questo documento analizza tecnicamente come il framework Apache Spark gestisce la parallelizzazione degli algoritmi implementati nel backend, senza la necessità di codice multi-thread esplicito.

## 1. Architettura di Runtime: `local[*]`

La sessione Spark è inizializzata con `.master("local[*]")`. Ecco cosa implica a livello di sistema:

*   **JVM Unica, Thread Multipli**: In modalità `local`, Driver ed Executor risiedono nella stessa Java Virtual Machine process.
*   **Parallelismo dei Task**: Il simbolo `[*]` istruisce Spark a creare tanti thread "worker" quanti sono i core logici della CPU host.
*   **Task Scheduling**: Quando viene lanciata un'azione (es. `fit()`, `count()`), il DAG Scheduler di Spark suddivide il lavoro in *Stage* e *Task*. Ogni Task opera su una singola partizione di dati. Se hai 8 core e il DataFrame ha 8 partizioni, 8 Task vengono eseguiti simultaneamente.

---

## 2. Il Grafo di Esecuzione (DAG) e Lazy Evaluation

Spark non esegue riga per riga come Python standard. Costruisce un **DAG (Directed Acyclic Graph)**.

1.  **Lazy Transformations**: Operazioni come `filter`, `map`, `withColumn` o lo step di pipeline `Tokenizer` non calcolano nulla subito. Aggiungono solo nodi al grafo logico.
2.  **Action Trigger**: Solo quando viene chiamata un'azione (es. `lda.fit()`, `collect()`), Spark:
    *   Ottimizza il piano di esecuzione (Catalyst Optimizer).
    *   Fonde più operazioni in un unico passaggio (Pipelining).
    *   Esegue i calcoli in parallelo.

---

## 3. Topic Modeling (LDA): Parallelismo Iterativo

L'algoritmo **Latent Dirichlet Allocation (LDA)** è computazionalmente pesante, ma Spark usa una versione ottimizzata chiamata **Online Variational Bayes**.

### Ciclo di Parallelizzazione
L'LDA è iterativo (`maxIter=20`). Ad ogni iterazione:
1.  **Broadcast**: I parametri globali del modello (distribuzione vocabolo-topic) vengono inviati a tutti i thread.
2.  **Map Distribuito (Parallel)**:
    *   Ogni thread prende un batch di documenti (recensioni negative).
    *   Calcola le variabili latenti locali per quei documenti (inferenza variazionale).
    *   Calcola i "sufficient statistics" (aggregati parziali necessari per aggiornare il modello).
3.  **Reduce/Aggregate (Shuffle)**:
    *   Le statistiche parziali di tutti i thread vengono aggregate (sommate).
4.  **Update**:
    *   Il driver aggiorna il modello globale con le somme aggregate.
    *   Il nuovo modello viene ritrasmesso per l'iterazione successiva.

### Persistenza (`persist`)
Nel codice, `df_neg.persist(StorageLevel.MEMORY_AND_DISK)` è cruciale.
*   Senza persistenza, Spark ricalcolerebbe l'intera pipeline di preprocessing (tokenizzazione, stop words) ad *ogni singola iterazione* dell'LDA.
*   Con persistenza, i vettori trasformati vengono salvati in RAM (o disco se la RAM finisce), permettendo all'LDA di iterare molto più velocemente.

---

## 4. Clustering (K-Means): Map-Reduce Geometrico

Il K-Means in Spark è un esempio classico di algoritmo Map-Reduce iterativo.

### Preprocessing Parallelo
La funzione `preprocessing_hotel_features` fa un uso intensivo di `groupBy`.
*   **Map-Side Combine**: Prima di inviare dati attraverso la rete (o tra thread), Spark esegue aggregazioni parziali locale. Esempio: ogni thread calcola la somma parziale e il conteggio parziale dei voti per gli hotel che gestisce.
*   **Shuffle**: I risultati parziali vengono redistribuiti in modo che tutti i dati dello stesso `Hotel_Name` finiscano sullo stesso thread per il calcolo finale.

### Training K-Means
1.  **Broadcast Centroids**: Le coordinate dei `k` centroidi vengono copiate in ogni contesto di esecuzione.
2.  **Assignment Step (Parallelo)**: Ogni thread calcola la distanza quadratica euclidea tra le sue righe e i centroidi, assegnando il cluster ID. Questo è "imbarazzantemente parallelo" (nessuna comunicazione necessaria tra righe).
3.  **Update Step (Shuffle)**: Spark calcola la nuova media per ogni cluster sommando le coordinate di tutti i punti assegnati e dividendo per il conteggio.

---

## 5. Sentiment Analysis (Logistic Regression)

La **Logistic Regression** utilizza algoritmi di ottimizzazione convessa distribuiti (es. L-BFGS).

### Calcolo del Gradiente Distribuito
L'addestramento cerca di trovare i pesi ($w$) che minimizzano l'errore.
1.  **Gradient Computation**: Per calcolare la direzione in cui muovere $w$, bisogna analizzare tutti i dati.
2.  **Parallelismo**:
    *   Ogni thread calcola il gradiente parziale sul suo sottoinsieme di dati.
    *   Viene usata un'operazione primitiva `treeAggregate` per sommare i vettori gradiente in modo gerarchico ed efficiente.
3.  **Step**: Il driver aggiorna $w$ usando il gradiente aggregato e distribuisce il nuovo $w$.

### Vettorizzazione (`HashingTF` vs `CountVectorizer`)
La pipeline usa `CountVectorizer`. Questo richiede un passaggio extra:
1.  **Passaggio 1 (Parallelo + Reduce)**: Scansiona tutto il testo per costruire il vocabolario (map: conta parole locali -> reduce: somma conteggi globali).
2.  **Passaggio 2 (Parallelo)**: Rilegge il testo e lo converte in vettori sparsi usando la mappa di indici costruita al passo 1.

---

## Conclusione

La parallelizzazione in questo progetto è:
1.  **Dati-Centrica**: I dati (DataFrame) sono partizionati.
2.  **Trasparente**: Non gestiamo lock, semafori o thread pool.
3.  **Scalabile**: Se spostassimo questo codice da un PC locale a un cluster AWS con 100 nodi, funzionerebbe senza cambiare una riga di logica, semplicemente cambiando `.master(...)` in configurazione.
