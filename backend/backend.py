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
    # ALGORITMO: CLUSTERING (K-Means) - FEATURE-BASED
    # ---------------------------------------------------------
    
    def preprocessing_hotel_features(self, df_hotel):
        """
        Feature Engineering per clustering intelligente degli hotel.
        Calcola 4 categorie di features:
        1. Performance Metrics (voto, popolarità)
        2. Sentiment Balance (ratio pos/neg)
        3. Audience Diversity (nazionalità, tipo viaggio)
        4. Problem Indicators (costruzioni, pulizia, staff)
        """
        from pyspark.sql.functions import length, when, sum as spark_sum, countDistinct
        
        # Group by Hotel_Name per aggregare tutte le recensioni
        df_features = df_hotel.groupBy("Hotel_Name", "lat", "lng") \
            .agg(
                # === 1. PERFORMANCE METRICS ===
                avg("Reviewer_Score").alias("voto_medio"),
                count("Reviewer_Score").alias("num_recensioni"),
                
                # % recensioni eccellenti (score >= 9)
                (spark_sum(when(col("Reviewer_Score") >= 9, 1).otherwise(0)) / count("*") * 100).alias("perc_eccellenti"),
                
                # % recensioni negative (score < 6)
                (spark_sum(when(col("Reviewer_Score") < 6, 1).otherwise(0)) / count("*") * 100).alias("perc_negative"),
                
                # === 2. SENTIMENT BALANCE ===
                # Ratio lunghezza media positive vs negative
                (avg(length(col("Positive_Review"))) / (avg(length(col("Negative_Review"))) + 1)).alias("ratio_pos_neg"),
                
                # Lunghezza media totale recensioni
                avg(length(col("Positive_Review")) + length(col("Negative_Review"))).alias("avg_review_length"),
                
                # === 3. AUDIENCE DIVERSITY ===
                # Numero nazionalità distinte
                countDistinct("Reviewer_Nationality").alias("num_nazionalita"),
                
                # === 4. PROBLEM INDICATORS ===
                # % recensioni con menzioni costruzione
                (spark_sum(when(col("Negative_Review").rlike("(?i)construction|renovation|works|noise"), 1).otherwise(0)) / count("*") * 100).alias("menzioni_costruzione"),
                
                # % recensioni con menzioni pulizia
                (spark_sum(when(col("Negative_Review").rlike("(?i)dirty|clean|hygiene"), 1).otherwise(0)) / count("*") * 100).alias("menzioni_pulizia"),
                
                # % recensioni con menzioni staff
                (spark_sum(when(col("Negative_Review").rlike("(?i)staff|service|reception|rude"), 1).otherwise(0)) / count("*") * 100).alias("menzioni_staff")
            )
        
        return df_features
    
    def esegui_clustering_hotel(self, df_hotel, k=4):
        """
        K-Means clustering basato su CARATTERISTICHE degli hotel (non geografiche).
        Identifica gruppi significativi: Premium, Budget, Hidden Gems, ecc.
        
        Args:
            df_hotel: DataFrame con recensioni
            k: numero di cluster
            
        Returns:
            dict con:
            - df_clustered: Hotel con cluster assignment
            - features_per_cluster: Medie features per ogni cluster
            - cluster_names: Interpretazione cluster
        """
        from pyspark.ml.feature import VectorAssembler, StandardScaler
        
        # 1. Calcola features
        df_features = self.preprocessing_hotel_features(df_hotel)
        
        # 2. Seleziona features numeriche per clustering (escludi lat/lng)
        feature_cols = [
            "voto_medio", "num_recensioni", "perc_eccellenti", "perc_negative",
            "ratio_pos_neg", "avg_review_length", "num_nazionalita",
            "menzioni_costruzione", "menzioni_pulizia", "menzioni_staff"
        ]
        
        # 3. Assembla feature vector
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
        df_assembled = assembler.transform(df_features)
        
        # 4. Normalizza features (importante per K-Means!)
        scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
        scaler_model = scaler.fit(df_assembled)
        df_scaled = scaler_model.transform(df_assembled)
        
        # 5. K-Means
        kmeans = KMeans(k=k, seed=42, maxIter=20, featuresCol="features", predictionCol="cluster")
        model = kmeans.fit(df_scaled)
        df_clustered = model.transform(df_scaled)
        
        # 6. Calcola statistiche per cluster
        cluster_stats = df_clustered.groupBy("cluster") \
            .agg(
                count("*").alias("num_hotel"),
                avg("voto_medio").alias("avg_voto"),
                avg("num_recensioni").alias("avg_recensioni"),
                avg("perc_eccellenti").alias("avg_eccellenti"),
                avg("menzioni_costruzione").alias("avg_problemi")
            ) \
            .orderBy("cluster")
        
        # 7. Interpretazione automatica cluster
        cluster_interpretations = self._interpreta_cluster(cluster_stats.toPandas())
        
        return {
            'df_clustered': df_clustered,
            'cluster_stats': cluster_stats,
            'cluster_names': cluster_interpretations,
            'model': model,
            'feature_cols': feature_cols
        }
    
    def _interpreta_cluster(self, stats_df):
        """
        Assegna nomi interpretativi ai cluster basandosi sulle statistiche.
        """
        interpretations = {}
        
        for _, row in stats_df.iterrows():
            cluster_id = int(row['cluster'])
            voto = row['avg_voto']
            recensioni = row['avg_recensioni']
            eccellenti = row['avg_eccellenti']
            problemi = row['avg_problemi']
            
            # Logica interpretativa
            if voto >= 8.5 and recensioni > 500:
                nome = "🏆 Premium Hotels"
            elif voto >= 8.0 and recensioni < 200:
                nome = "💎 Hidden Gems"
            elif voto < 7.0 or problemi > 15:
                nome = "📉 Budget/Problems"
            elif recensioni > 800:
                nome = "🌟 Popular Mixed"
            else:
                nome = f"📊 Cluster {cluster_id}"
            
            interpretations[cluster_id] = nome
        
        return interpretations

    # ---------------------------------------------------------
    # 3. ALGORITMO: TOPIC MODELING (LDA)
    # ---------------------------------------------------------

    def esegui_topic_modeling(self, df_hotel, num_topics=3, evaluate_stability=True, top_terms=10):
        """
        Estrae topic latenti dalle recensioni NEGATIVE usando LDA.

        Upgrade (passo 6):
        - Valuta la stabilità dei topic ripetendo LDA con seed diversi
        - Calcola una metrica: Jaccard similarity sui top-terms (0..1)
        - Mantiene lo stesso vocabolario/features per rendere il confronto corretto
        """
        from pyspark.sql.functions import col, length, lower, lit
        from pyspark.storagelevel import StorageLevel

        from pyspark.ml import Pipeline
        from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
        from pyspark.ml.clustering import LDA

        # ---------- 1) Filtri negative review "significative" ----------
        df_neg = df_hotel.filter(
            (col("Negative_Review") != "No Negative") &
            col("Negative_Review").isNotNull() &
            (length(col("Negative_Review")) > 30)
        )

        df_neg = df_neg.withColumn("review_lower", lower(col("Negative_Review")))

        df_neg = df_neg.persist(StorageLevel.MEMORY_AND_DISK)
        num_reviews = df_neg.count()

        # ---------- 2) Preprocessing testo (tokenizer robusto unicode) ----------
        tokenizer = RegexTokenizer(
            inputCol="review_lower",
            outputCol="words",
            pattern="\\p{L}{3,}",
            gaps=False
        )

        # Stopwords base + (consigliato) stopwords di dominio
        domain_stop = ["hotel", "room", "rooms", "stay", "staff", "place", "night", "nights"]
        remover = StopWordsRemover(
            inputCol="words",
            outputCol="filtered",
            stopWords=StopWordsRemover.loadDefaultStopWords("english") + domain_stop
        )

        # minDF deve essere int (>= 15 documenti)
        cv = CountVectorizer(
            inputCol="filtered",
            outputCol="features",
            vocabSize=1500,
            minDF=15
        )

        # ---------- 3) Fissa vocabolario/features UNA sola volta ----------
        feat_pipeline = Pipeline(stages=[tokenizer, remover, cv])
        feat_model = feat_pipeline.fit(df_neg)
        df_feat = feat_model.transform(df_neg).select("features")

        cv_model = feat_model.stages[2]
        vocab = cv_model.vocabulary

        # Helper: estrazione top-terms (lista di liste) da lda_model
        def extract_topic_terms(lda_model, vocab, top_terms):
            topics = lda_model.describeTopics(maxTermsPerTopic=top_terms).collect()
            out = []
            for r in topics:
                inds = r["termIndices"]
                out.append([vocab[int(i)] for i in inds if int(i) < len(vocab)])
            return out  # list of topics -> list of terms

        # Helper: Jaccard
        def jaccard(a, b):
            sa, sb = set(a), set(b)
            if len(sa) == 0 and len(sb) == 0:
                return 1.0
            if len(sa) == 0 or len(sb) == 0:
                return 0.0
            return len(sa.intersection(sb)) / len(sa.union(sb))

        # Helper: matching greedy topic-to-topic tra due run
        def greedy_match(topics_A, topics_B):
            used = set()
            matches = []
            for i, ta in enumerate(topics_A):
                best_j, best_sim = None, -1.0
                for j, tb in enumerate(topics_B):
                    if j in used:
                        continue
                    sim = jaccard(ta, tb)
                    if sim > best_sim:
                        best_sim = sim
                        best_j = j
                used.add(best_j)
                matches.append((i, best_j, best_sim))
            return matches  # (topicA, topicB, sim)

        # ---------- 4) Fit modello "principale" ----------
        lda_main = LDA(k=num_topics, maxIter=20, optimizer="online", seed=42)
        lda_model_main = lda_main.fit(df_feat)

        topics_data_main = lda_model_main.describeTopics(maxTermsPerTopic=top_terms)
        log_likelihood = float(lda_model_main.logLikelihood(df_feat))
        log_perplexity = float(lda_model_main.logPerplexity(df_feat))

        stability = None
        stability_table = None
        compare_topics = None

        # ---------- 5) Passo 6: stability check (seed diverso) ----------
        if evaluate_stability:
            lda_alt = LDA(k=num_topics, maxIter=20, optimizer="online", seed=99)
            lda_model_alt = lda_alt.fit(df_feat)

            terms_main = extract_topic_terms(lda_model_main, vocab, top_terms)
            terms_alt = extract_topic_terms(lda_model_alt, vocab, top_terms)

            matches = greedy_match(terms_main, terms_alt)

            # costruisci risultati: per-topic + overall
            per_topic = []
            for a, b, sim in matches:
                per_topic.append({
                    "topic_main": int(a),
                    "topic_alt": int(b),
                    "jaccard": float(sim),
                    "main_terms": terms_main[a],
                    "alt_terms": terms_alt[b]
                })

            overall = sum(x["jaccard"] for x in per_topic) / len(per_topic)

            stability = float(overall)
            stability_table = per_topic  # lista di dict pronti per pandas
            compare_topics = {
                "seed_main": 42,
                "seed_alt": 99
            }

        df_neg.unpersist()

        return {
            "topics_data": topics_data_main,
            "vocab": vocab,
            "num_reviews": num_reviews,
            "log_perplexity": log_perplexity,
            "log_likelihood": log_likelihood,
            # passo 6 output
            "stability": stability,
            "stability_table": stability_table,
            "compare_topics": compare_topics
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
        from pyspark.sql.functions import when, col, avg, count, stddev, lower, sqrt, round, lit
        
        df_viaggi = df_hotel.withColumn("tags_lc", lower(col("Tags")))
        # Ordine categorie: Solo / Family / Group / Couple / Other
        df_viaggi = df_viaggi.withColumn(
            "tipo_viaggio",
            when(col("tags_lc").contains("solo traveler"), "🚶 Solo")
            .when(col("tags_lc").contains("family"), "👨‍👩‍👧‍👦 Famiglia")
            .when(col("tags_lc").contains("group"), "👥 Gruppo")
            .when(col("tags_lc").contains("couple"), "👫 Coppia")
            .otherwise("❓ Altro")
        )
        
        agg = (
            df_viaggi.groupBy("tipo_viaggio")
            .agg(
                avg("Reviewer_Score").alias("voto_medio"),
                count("*").alias("num_recensioni"),
                stddev("Reviewer_Score").alias("deviazione_std"),
            )
            .filter(col("num_recensioni") > 50)
        )
        # CI 95% circa: mean ± 1.96 * (std/sqrt(n))
        agg = agg.withColumn("se", col("deviazione_std") / sqrt(col("num_recensioni")))
        agg = agg.withColumn("ci95", lit(1.96) * col("se"))

        # Pulizia output
        agg = (
            agg.withColumn("voto_medio", round(col("voto_medio"), 3))
            .withColumn("deviazione_std", round(col("deviazione_std"), 3))
            .withColumn("ci95", round(col("ci95"), 3))
            .orderBy(col("voto_medio").desc())
        )

        return agg.drop("se")

    def query_lunghezza_recensioni(self, df_hotel):
        """
        Query 4: 'Asimmetria Emotiva' (Emotional Asymmetry).
        Analizza se la delusione genera più testo della soddisfazione.

        Output per bucket di score:
        - lunghezze medie (neg/pos)
        - differenza (neg - pos)
        - negativity_ratio (solo se avg_positive_length è sufficientemente > 0)
        - % recensioni con testo negativo/positivo presente
        """
        from pyspark.sql.functions import avg, when, col, lit, count, round

        df_bucket = (
            df_hotel
            .withColumn(
                "bucket_id",
                when(col("Reviewer_Score") < 5.0, lit(0))
                .when((col("Reviewer_Score") >= 5.0) & (col("Reviewer_Score") < 7.5), lit(1))
                .when((col("Reviewer_Score") >= 7.5) & (col("Reviewer_Score") < 9.0), lit(2))
                .otherwise(lit(3))
            )
            .withColumn(
                "score_bucket",
                when(col("Reviewer_Score") < 5.0, "😠 < 5.0 (Arrabbiato)")
                .when((col("Reviewer_Score") >= 5.0) & (col("Reviewer_Score") < 7.5), "😐 5.0-7.5 (Deluso)")
                .when((col("Reviewer_Score") >= 7.5) & (col("Reviewer_Score") < 9.0), "🙂 7.5-9.0 (Soddisfatto)")
                .otherwise("😍 > 9.0 (Felice)")
            )
        )

        agg = (
            df_bucket.groupBy("bucket_id", "score_bucket")
            .agg(
                avg("Review_Total_Negative_Word_Counts").alias("avg_negative_length"),
                avg("Review_Total_Positive_Word_Counts").alias("avg_positive_length"),

                # % review con testo presente (proxy: word_count > 0)
                (count(when(col("Review_Total_Negative_Word_Counts") > 0, 1)) / count("*") * 100).alias("pct_has_negative"),
                (count(when(col("Review_Total_Positive_Word_Counts") > 0, 1)) / count("*") * 100).alias("pct_has_positive"),

                count("*").alias("num_reviews")
            )
        )

        # Differenza (molto interpretabile)
        agg = agg.withColumn(
            "delta_len_neg_minus_pos",
            col("avg_negative_length") - col("avg_positive_length")
        )

        # Ratio solo se il denominatore è “abbastanza grande” (evita artefatti)
        agg = agg.withColumn(
            "negativity_ratio",
            when(col("avg_positive_length") >= 3.0, col("avg_negative_length") / col("avg_positive_length"))
            .otherwise(lit(None))
        )

        # Arrotondamenti per UI
        result = (
            agg.withColumn("avg_negative_length", round(col("avg_negative_length"), 2))
            .withColumn("avg_positive_length", round(col("avg_positive_length"), 2))
            .withColumn("delta_len_neg_minus_pos", round(col("delta_len_neg_minus_pos"), 2))
            .withColumn("negativity_ratio", round(col("negativity_ratio"), 3))
            .withColumn("pct_has_negative", round(col("pct_has_negative"), 2))
            .withColumn("pct_has_positive", round(col("pct_has_positive"), 2))
            .orderBy("bucket_id")
            .drop("bucket_id")
        )

        return result

    def query_affidabilita_voto(self, df_hotel):
        """
        Query 5: 'Affidabilità del Voto' (Data Consistency).
        Misura la deviazione standard dei voti per capire se il punteggio medio è rappresentativo.

        Nota:
        - Evito first(Average_Score) perché non deterministico.
        """
        from pyspark.sql.functions import stddev, avg, count, col, round

        result = (
            df_hotel.groupBy("Hotel_Name")
            .agg(
                avg("Average_Score").alias("avg_hotel_score"),                 # aspettativa "ufficiale" (robusta)
                avg("Reviewer_Score").alias("mean_reviewer_score"),            # media reale
                stddev("Reviewer_Score").alias("stddev_reviewer_score"),       # dispersione (campionaria)
                count("*").alias("num_reviews")
            )
            .filter(col("num_reviews") >= 100)
        )

        # CV = std/mean (utile per confrontare hotel con medie diverse)
        result = result.withColumn(
            "cv_score",
            round(col("stddev_reviewer_score") / col("mean_reviewer_score"), 4)
        )

        # pulizia output
        result = (
            result.withColumn("avg_hotel_score", round(col("avg_hotel_score"), 2))
                .withColumn("mean_reviewer_score", round(col("mean_reviewer_score"), 2))
                .withColumn("stddev_reviewer_score", round(col("stddev_reviewer_score"), 2))
                .orderBy(col("stddev_reviewer_score").desc())
        )

        return result

    def query_hotel_rischiosi(self, df_hotel):
        """
        Query 6: 'Hotel Rischiosi' (High Risk, High Reward?).
        Individua hotel con media alta (>8) ma con una percentuale preoccupante di recensioni disastrose (<=4).
        
        Logica:
        - Calcola la % di recensioni <= 4.0 per ogni hotel.
        - Filtra quelli con media >= 8.0 ma % disastri > 5%.
        """
        from pyspark.sql.functions import (
            avg, count, when, col, round, log1p, percentile_approx
        )

        # 1) Aggregazione per hotel
        result = (
            df_hotel.groupBy("Hotel_Name")
            .agg(
                avg("Average_Score").alias("avg_hotel_score"),            # aspettativa (stabile e deterministica)
                avg("Reviewer_Score").alias("mean_reviewer_score"),      # realtà media
                count("*").alias("total_reviews"),
                count(when(col("Reviewer_Score") <= 4.0, 1)).alias("disaster_count"),
                percentile_approx("Reviewer_Score", 0.05).alias("p05_score")  # coda bassa (robusta)
            )
        )

        # 2) Percentuale disastri
        result = result.withColumn(
            "disaster_pct",
            round((col("disaster_count") / col("total_reviews")) * 100, 2)
        )

        # 3) Indice rischio (ranking più solido)
        result = result.withColumn(
            "risk_index",
            round(col("disaster_pct") * log1p(col("total_reviews")), 2)
        )

        # 4) Filtri: significatività + “apparenza ottima” + rischio
        result = (
            result.filter(
                (col("total_reviews") >= 50) &
                (col("avg_hotel_score") >= 8.0) &        # hotel che “sembrano ottimi”
                (col("disaster_pct") >= 5.0)
            )
            .orderBy(col("risk_index").desc(), col("disaster_pct").desc())
        )

        return result

    def query_expectation_gap(self, df_hotel):
        """
        Query 7: 'Expectation Gap' Analysis.
        Analizza la differenza tra Aspettativa (Average_Score) e Realtà (Reviewer_Score).

        Logica:
        1. Calcola il Gap = Reviewer_Score - Average_Score per ogni recensione.
        2. Divide in bucket di 'Prestigio' (basato su Average_Score).
        3. Misura quanto spesso le aspettative vengono deluse (Gap < 0) e con che intensità.
        """

        from pyspark.sql.functions import (
            avg, col, when, count, round, lit, coalesce
        )

        # 1) Calcolo del Gap (Realtà - Aspettativa)
        df_gap = df_hotel.withColumn("gap", col("Reviewer_Score") - col("Average_Score"))

        # 2) Bucket di Prestigio + bucket_id numerico per ordinamento logico
        df_gap = df_gap.withColumn(
            "bucket_id",
            when(col("Average_Score") < 7.5, lit(0))
            .when((col("Average_Score") >= 7.5) & (col("Average_Score") < 8.5), lit(1))
            .when((col("Average_Score") >= 8.5) & (col("Average_Score") < 9.2), lit(2))
            .otherwise(lit(3))
        ).withColumn(
            "expectation_bucket",
            when(col("Average_Score") < 7.5, "🥉 Economico (< 7.5)")
            .when((col("Average_Score") >= 7.5) & (col("Average_Score") < 8.5), "🥈 Standard (7.5-8.5)")
            .when((col("Average_Score") >= 8.5) & (col("Average_Score") < 9.2), "🥇 Premium (8.5-9.2)")
            .otherwise("💎 Luxury (> 9.2)")
        )

        # 3) Aggregazione per Bucket
        result = (
            df_gap.groupBy("bucket_id", "expectation_bucket")
            .agg(
                avg("gap").alias("avg_gap"),

                # Percentuale di Gap Negativi (Delusioni)
                (count(when(col("gap") < 0, 1)) / count("*") * 100).alias("pct_delusioni"),

                # Intensità media della delusione (solo sui gap negativi)
                avg(when(col("gap") < 0, col("gap"))).alias("intensita_delusione_media"),

                count("*").alias("num_reviews"),
            )
            .orderBy("bucket_id")
            .drop("bucket_id")
        )

        # 4) Pulizia output per visualizzazione (Streamlit-friendly)
        #    - round per leggibilità
        #    - coalesce per evitare null (caso: nessun gap negativo nel bucket)
        result = (
            result.withColumn("avg_gap", round(col("avg_gap"), 3))
                  .withColumn("pct_delusioni", round(col("pct_delusioni"), 2))
                  .withColumn(
                  "intensita_delusione_media",
                  round(coalesce(col("intensita_delusione_media"), lit(0.0)), 3)
              )
        )
        return result

