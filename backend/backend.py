import os
import sys

# Configurazione ambiente per PySpark su Windows
os.environ['JAVA_HOME'] = r"C:\Program Files\Microsoft\jdk-17.0.18.8-hotspot"
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
# JDK_JAVA_OPTIONS rimosse perché Java 17 non necessita dei flag per Java 25

from pyspark.sql import SparkSession

from pyspark.sql.functions import col, lower, udf, regexp_replace, split, avg, count, to_date, month, when
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans, LDA
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, ClusteringEvaluator

class GestoreBigData:
    """
    Classe singleton per gestire la sessione Spark e le operazioni di analisi dati.
    Mantiene il contesto "semplice e pulito" come richiesto.
    """
    _spark = None

    @staticmethod
    def get_spark_session():
        """
        Inizializza o restituisce la sessione Spark esistente.
        Configurata per utilizzare le risorse locali.
        """
        if GestoreBigData._spark is None:
            GestoreBigData._spark = SparkSession.builder \
                .appName("ProgettoBigDataHotel") \
                .master("local[*]") \
                .config("spark.driver.memory", "4g") \
                .config("spark.executor.memory", "4g") \
                .config("spark.memory.offHeap.enabled", "true") \
                .config("spark.memory.offHeap.size", "1g") \
                .config("spark.sql.shuffle.partitions", "8") \
                .getOrCreate()
        return GestoreBigData._spark

    def carica_dati_hotel(self, percoso_file):
        """
        Carica il dataset delle recensioni degli hotel.
        Effettua una pulizia base dei dati.
        """
        spark = self.get_spark_session()
        # Caricamento con header e inferenza schema
        df = spark.read.csv(percoso_file, header=True, inferSchema=True)
        
        # Pulizia: Rimuoviamo righe con valori nulli nelle colonne critiche
        df_pulito = df.na.drop(subset=["Hotel_Name", "Review_Date", "Negative_Review", "Positive_Review"])
        
        # FIX: Alcuni valori numerici sono stringhe 'NA' invece di null
        # Usiamo na.replace per sostituirli con None (più sicuro del confronto diretto)
        df_pulito = df_pulito.na.replace("NA", None)
        
        # Casting colonne numeriche necessario se l'inferenza ha fallito parzialmente
        df_pulito = df_pulito.withColumn("lat", col("lat").cast("double")) \
                             .withColumn("lng", col("lng").cast("double")) \
                             .withColumn("Average_Score", col("Average_Score").cast("double")) \
                             .withColumn("Total_Number_of_Reviews", col("Total_Number_of_Reviews").cast("int")) \
                             .withColumn("Reviewer_Score", col("Reviewer_Score").cast("double"))
        
        # Dopo il casting, rimuoviamo righe con valori nulli in colonne critiche per clustering
        df_pulito = df_pulito.na.drop(subset=["lat", "lng", "Average_Score"])

        return df_pulito

    # ---------------------------------------------------------
    # ALGORITMO: SENTIMENT ANALYSIS (Logistic Regression)
    # ---------------------------------------------------------
    def allena_sentiment_hotel(self, df_hotel):
        """
        Addestra un modello di Sentiment Analysis usando il dataset Hotel stesso.
        Strategia: Usa Reviewer_Score per creare le label
        - Score >= 7.5 → Sentiment Positivo (label=1)
        - Score < 7.5 → Sentiment Negativo (label=0)
        
        NUOVO: Restituisce anche statistiche per visualizzazione
        """
        from pyspark.sql.functions import concat_ws, when
        from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
        from pyspark.ml.classification import LogisticRegression
        from pyspark.ml import Pipeline
        from pyspark.ml.evaluation import MulticlassClassificationEvaluator
        
        # Prepara i dati: combina le recensioni
        df_prep = df_hotel.withColumn("review", concat_ws(" ", col("Negative_Review"), col("Positive_Review")))
        
        # Crea label: 1 se score >= 7.5 (positivo), 0 altrimenti (negativo)
        df_prep = df_prep.withColumn("label", when(col("Reviewer_Score") >= 7.5, 1.0).otherwise(0.0))
        
        # Filtra recensioni vuote o troppo corte
        df_prep = df_prep.filter("length(review) > 20")
        
        # Campiona il 30% dei dati per velocità (opzionale su macchine con poca RAM)
        df_sampled = df_prep.sample(False, 0.3, seed=42)
        
        # Pipeline ML
        tokenizer = Tokenizer(inputCol="review", outputCol="words")
        remover = StopWordsRemover(inputCol="words", outputCol="filtered")
        cv = CountVectorizer(inputCol="filtered", outputCol="rawFeatures", vocabSize=800, minDF=5.0)
        idf = IDF(inputCol="rawFeatures", outputCol="features")
        lr = LogisticRegression(maxIter=10, regParam=0.01)
        
        pipeline = Pipeline(stages=[tokenizer, remover, cv, idf, lr])
        
        # Split train/test
        train_data, test_data = df_sampled.randomSplit([0.8, 0.2], seed=42)
        train_data.cache()
        
        # NUOVO: Conta distribuzione label nel training set
        train_label_dist = train_data.groupBy("label").count().collect()
        train_label_counts = {int(row['label']): row['count'] for row in train_label_dist}
        
        # Addestramento
        modello = pipeline.fit(train_data)
        
        # Valutazione sul test set
        predizioni_test = modello.transform(test_data)
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predizioni_test)
        
        # NUOVO: Conta distribuzione predizioni sul test set
        test_pred_dist = predizioni_test.groupBy("prediction").count().collect()
        test_pred_counts = {int(row['prediction']): row['count'] for row in test_pred_dist}
        
        # NUOVO: Estrai esempi di recensioni classificate (per debugging/visualizzazione)
        esempi = predizioni_test.select("review", "label", "prediction", "probability") \
                                .limit(10).toPandas()
        
        # Conta totale recensioni
        total_reviews = df_sampled.count()
        
        # Restituisci modello + statistiche
        return {
            'modello': modello,
            'accuracy': accuracy,
            'train_label_counts': train_label_counts,  # {0: count_neg, 1: count_pos}
            'test_pred_counts': test_pred_counts,       # {0: count_neg, 1: count_pos}
            'esempi_predizioni': esempi,                # DataFrame pandas con esempi
            'total_reviews': total_reviews
        }

    # ---------------------------------------------------------
    # ALGORITMO: CLUSTERING (K-Means)
    # ---------------------------------------------------------
    def esegui_clustering_hotel(self, df_hotel, k=5):
        """
        Raggruppa gli hotel in base a caratteristiche geografiche e di qualità.
        Features usate: lat, lng, Average_Score.
        """
        # Assembla le feature in un unico vettore
        assembler = VectorAssembler(
            inputCols=["lat", "lng", "Average_Score"],
            outputCol="features"
        )
        
        kmeans = KMeans().setK(k).setSeed(1)
        pipeline = Pipeline(stages=[assembler, kmeans])
        
        modello_km = pipeline.fit(df_hotel)
        risultati = modello_km.transform(df_hotel)
        
        # Valutazione (Silhouette Score)
        evaluator = ClusteringEvaluator()
        silhouette = evaluator.evaluate(risultati)
        
        return risultati, silhouette

    # ---------------------------------------------------------
    # 3. ALGORITMO: TOPIC MODELING (LDA)
    # ---------------------------------------------------------
    def esegui_topic_modeling(self, df_hotel, num_topics=3):
        """
        Estrae topic latenti dalle recensioni NEGATIVE usando LDA.
        MIGLIORAMENTI:
        - Filtra recensioni troppo corte o generiche
        - Aumenta vocabolario e numero di termini per topic
        - Restituisce pesi dei termini e metriche di qualità
        """
        from pyspark.sql.functions import length
        
        # Filtra recensioni negative significative (non vuote e non generiche)
        df_neg = df_hotel.filter(
            (col("Negative_Review") != "No Negative") & 
            (col("Negative_Review").isNotNull()) &
            (length(col("Negative_Review")) > 30)  # Almeno 30 caratteri per evitare recensioni troppo brevi
        )
        
        # Conta quante recensioni negative stiamo analizzando
        num_reviews = df_neg.count()
        
        # FIX: Pulizia testo AGGRESSIVA - rimuovi punteggiatura, simboli e numeri
        from pyspark.sql.functions import regexp_replace, lower
        
        # Step 1: Converti in minuscolo
        df_neg = df_neg.withColumn("review_lower", lower(col("Negative_Review")))
        
        # Step 2: Rimuovi TUTTO tranne lettere e spazi (elimina *, -, numeri, ecc.)
        df_neg = df_neg.withColumn("Negative_Review_Clean", 
                                   regexp_replace(col("review_lower"), r'[^a-z\s]', ' '))
        
        # Step 3: Rimuovi spazi multipli
        df_neg = df_neg.withColumn("Negative_Review_Clean",
                                   regexp_replace(col("Negative_Review_Clean"), r'\s+', ' '))
        
        
        # Pipeline preprocessing con StopWords CUSTOM
        tokenizer = Tokenizer(inputCol="Negative_Review_Clean", outputCol="words")
        
        # SOLUZIONE: Aggiungi manualmente i simboli alle stopwords
        custom_stopwords = StopWordsRemover.loadDefaultStopWords("english") + [
            "*", "**", "***", "****", "*****",  # Asterischi
            "-", "--", "---", "----",             # Trattini
            "!", "!!", "!!!",                      # Esclamativi
            ".", "..", "...",                      # Punti
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",  # Numeri
        ]
        
        remover = StopWordsRemover(inputCol="words", outputCol="filtered", stopWords=custom_stopwords)
        
        # Aumentato vocabSize per catturare più termini specifici
        # minDF=10 significa: ignora parole che appaiono in meno di 10 documenti
        cv = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=1500, minDF=15.0)
        
        # LDA con più iterazioni per convergenza migliore
        lda = LDA(k=num_topics, maxIter=20, optimizer="online", seed=42)
        
        # Pipeline semplice - solo CV e LDA
        pipeline = Pipeline(stages=[tokenizer, remover, cv, lda])
        
        # Fit sui dati
        modello_lda = pipeline.fit(df_neg)
        risultati = modello_lda.transform(df_neg)
        
        # Estrazione modelli e metriche
        cv_model = modello_lda.stages[2]
        lda_model = modello_lda.stages[3]
        vocab = cv_model.vocabulary
        
        # Descrivi topic con PIÙ termini (10 invece di 5) e restituisci anche i pesi
        topics_data = lda_model.describeTopics(maxTermsPerTopic=10)
        
        # Calcola perplexity come metrica di qualità (più basso = migliore)
        log_likelihood = lda_model.logLikelihood(risultati)
        log_perplexity = lda_model.logPerplexity(risultati)
        
        # Restituisci tutto
        return {
            'risultati': risultati,
            'topics_data': topics_data,
            'vocab': vocab,
            'num_reviews': num_reviews,
            'log_perplexity': log_perplexity,
            'log_likelihood': log_likelihood
        }

    # ---------------------------------------------------------
    # 4. QUERY AVANZATE (Nuove richieste)
    # ---------------------------------------------------------
    
    
    def query_nazionalita_critiche(self, df_hotel):
        """
        Query 1: 'The Grumpy Tourist'.
        Analizza quali nazionalità tendono dare voti più bassi o più alti.
        MIGLIORAMENTI: Calcola deviazione standard, min/max, e percentili
        """
        from pyspark.sql.functions import stddev, min as spark_min, max as spark_max, percentile_approx
        
        result = df_hotel.groupBy("Reviewer_Nationality") \
            .agg(
                avg("Reviewer_Score").alias("voto_medio"),
                count("Reviewer_Score").alias("num_recensioni"),
                stddev("Reviewer_Score").alias("deviazione_std"),
                spark_min("Reviewer_Score").alias("voto_min"),
                spark_max("Reviewer_Score").alias("voto_max")
            ) \
            .filter(col("num_recensioni") > 100) \
            .orderBy("voto_medio")
        
        return result

    def query_impatto_costruzioni(self, df_hotel):
        """
        Query 2: 'Construction Nightmare'.
        Analizza l'impatto dei lavori in corso sul voto dell'utente.
        MIGLIORAMENTI: Aggiungi campioni di recensioni e conteggio parole chiave
        """
        from pyspark.sql.functions import stddev
        
        # Creiamo una colonna flag se la recensione menziona lavori in corso
        df_costruzioni = df_hotel.withColumn(
            "has_construction", 
            col("Negative_Review").rlike("(?i)construction|renovation|works|hammering|drilling|noise|building")
        )
        
        # Statistiche aggregate
        stats = df_costruzioni.groupBy("has_construction") \
            .agg(
                avg("Reviewer_Score").alias("voto_medio"),
                count("Reviewer_Score").alias("totale"),
                stddev("Reviewer_Score").alias("deviazione_std")
            )
        
        return stats

    def query_coppie_vs_famiglie(self, df_hotel):
        """
        Query 3: 'Couple vs Family'.
        MIGLIORAMENTI: Categorie più complete e statistiche aggregate
        """
        from pyspark.sql.functions import stddev
        
        df_viaggi = df_hotel.withColumn(
            "tipo_viaggio",
            when(col("Tags").like("%Couple%"), "👫 Coppia")
            .when(col("Tags").like("%Family%"), "👨‍👩‍👧‍👦 Famiglia")
            .when(col("Tags").like("%Solo traveler%"), "🚶 Solo")
            .when(col("Tags").like("%Group%"), "👥 Gruppo")
            .otherwise("❓ Altro")
        )
        
        return df_viaggi.groupBy("tipo_viaggio") \
            .agg(
                avg("Reviewer_Score").alias("voto_medio"),
                count("Reviewer_Score").alias("num_recensioni"),
                stddev("Reviewer_Score").alias("deviazione_std")
            ) \
            .filter(col("num_recensioni") > 50) \
            .orderBy(col("voto_medio").desc())
