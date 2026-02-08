import streamlit as st
import pandas as pd
import sys
import os

# Aggiunge la cartella 'backend' al path per permettere l'import del modulo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from backend import GestoreBigData
from pyspark.sql.functions import avg, col, count # AGGIUNTI per la mappa

# Configurazione pagina
st.set_page_config(page_title="Analisi Big Data Hotel", layout="wide")

# Percorsi dei file (Hardcoded come da contesto ambiente utente)
PATH_HOTEL = r"f:\prog_big_1\_dati\Hotel_Reviews.csv"

# Inizializzazione Singleton Backend (Cachato per non riavviare Spark ad ogni reload)
@st.cache_resource
def get_backend():
    return GestoreBigData()

gestore = get_backend()

# Titolo e Descrizione
st.title("🏨 Recensioni Hotel")
st.markdown("""
Questa applicazione utilizza **Apache Spark** per analizzare un dataset di recensioni di hotel. 
Include algoritmi di Machine Learning per Sentiment Analysis, Clustering e Topic Modeling.
""")

# Sidebar per controlli globali
st.sidebar.header("Gestione Dati")

# Stato della sessione per mantenere i dati caricati in memoria (puntatori ai DF Spark)
if 'df_hotel' not in st.session_state:
    st.session_state.df_hotel = None

# Caricamento Dati
if st.sidebar.button("Carica Dataset"):
    with st.spinner("Inizializzazione Spark e caricamento dati in corso..."):
        try:
            st.session_state.df_hotel = gestore.carica_dati_hotel(PATH_HOTEL)
            st.success("Dati caricati con successo!")
        except Exception as e:
            st.error(f"Errore nel caricamento: {e}")

# Sidebar Navigation
page = st.sidebar.radio(
    "📍 Navigazione",
    ["📊 Esplorazione Dati", " Sentiment Analysis", "🗺️ Mappa & Clustering", 
     "📝 Topic Modeling", "🧠 Insight Avanzati"]
)
st.sidebar.markdown("---")


# PAGINA 1: ESPLORAZIONE DATI
if page == "📊 Esplorazione Dati":
    st.header("📊 Esplorazione Dati")
    
    if st.session_state.df_hotel:
        # === SEZIONE 1: STATISTICHE GENERALI ===
        st.subheader("📈 Statistiche Generali")
        
        # Calcola metriche
        num_hotel = st.session_state.df_hotel.count()
        avg_score = st.session_state.df_hotel.select("Average_Score").agg({"Average_Score": "avg"}).collect()[0][0]
        num_hotels_unique = st.session_state.df_hotel.select("Hotel_Name").distinct().count()
        
        # Layout a 3 colonne per metriche
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📝 Totale Recensioni", f"{num_hotel:,}")
        with col2:
            st.metric("⭐ Punteggio Medio", f"{avg_score:.2f}/10")
        with col3:
            st.metric("🏨 Hotel Unici", f"{num_hotels_unique:,}")
        
        st.markdown("")  # Spacing
        st.divider()
        
        # === SEZIONE 2: ANTEPRIMA DATASET ===
        st.subheader("🔍 Anteprima Dataset")
        st.caption("Visualizzazione delle prime 1000 recensioni")
        
        # Opzioni di visualizzazione
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption("Seleziona colonne da visualizzare:")
        with col2:
            show_all = st.checkbox("Mostra tutte le colonne", value=False)
        
        if show_all:
            st.dataframe(st.session_state.df_hotel.limit(200).toPandas(), use_container_width=True)
        else:
            # Mostra solo colonne rilevanti
            columns_to_show = ["Hotel_Name", "Reviewer_Score", "Reviewer_Nationality", "Positive_Review", "Negative_Review"]
            st.dataframe(
                st.session_state.df_hotel.select(columns_to_show).limit(200).toPandas(),
                use_container_width=True,
                height=400
            )
        
        st.markdown("")  # Spacing
    else:
        st.info("💡 Carica i dati dalla sidebar per iniziare l'esplorazione.")

# PAGINA 2: SENTIMENT ANALYSIS
elif page == " Sentiment Analysis":
    st.header(" Sentiment Analysis (Logistic Regression)")
    st.markdown("""
    Analizziamo il **sentiment** delle recensioni Hotel usando il **Reviewer_Score** come label:
    - **Score ≥ 7.5** → Sentiment Positivo ✅
    - **Score < 7.5** → Sentiment Negativo ❌
    
    Il modello impara a predire il sentiment basandosi sul testo delle recensioni.
    """)
    
    if st.session_state.df_hotel:
        if st.button("🚀 Addestra Modello Sentiment", type="primary"):
            with st.spinner("Addestramento in corso (può richiedere 1-2 minuti)..."):
                try:
                    result = gestore.allena_sentiment_hotel(st.session_state.df_hotel)
                    
                    # Unwrap results
                    modello = result['modello']
                    accuracy = result['accuracy']
                    train_label_counts = result['train_label_counts']
                    test_pred_counts = result['test_pred_counts']
                    esempi = result['esempi_predizioni']
                    total_reviews = result['total_reviews']
                    
                    st.success(f"✅ Modello addestrato con successo su **{total_reviews:,}** recensioni!")
                    
                    # Metriche principali
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🎯 Accuratezza", f"{accuracy:.2%}")
                    with col2:
                        st.metric("📊 Training Set", f"{sum(train_label_counts.values()):,}")
                    with col3:
                        balance_ratio = train_label_counts.get(1, 0) / max(train_label_counts.get(0, 1), 1)
                        st.metric("⚖️ Bilanciamento", f"{balance_ratio:.2f}", help="Ratio Pos/Neg nel training")
                    
                    st.divider()
                    
                    # GRAFICI DI VISUALIZZAZIONE
                    st.subheader("📊 Visualizzazione Training")
                    
                    col1, col2 = st.columns(2)
                    
                    # Grafico 1: Distribuzione Label Training
                    with col1:
                        st.markdown("**Distribuzione Label (Training Set)**")
                        import pandas as pd
                        train_df = pd.DataFrame({
                            'Sentiment': ['Negativo (< 7.5)', 'Positivo (≥ 7.5)'],
                            'Count': [train_label_counts.get(0, 0), train_label_counts.get(1, 0)]
                        })
                        st.bar_chart(train_df.set_index('Sentiment'), color="#FF6B6B")
                        st.caption(f"📉 Negativo: {train_label_counts.get(0, 0):,} | 📈 Positivo: {train_label_counts.get(1, 0):,}")
                    
                    # Grafico 2: Distribuzione Predizioni Test
                    with col2:
                        st.markdown("**Distribuzione Predizioni (Test Set)**")
                        test_df = pd.DataFrame({
                            'Sentiment': ['Negativo (Pred)', 'Positivo (Pred)'],
                            'Count': [test_pred_counts.get(0, 0), test_pred_counts.get(1, 0)]
                        })
                        st.bar_chart(test_df.set_index('Sentiment'), color="#4ECDC4")
                        st.caption(f"📉 Neg: {test_pred_counts.get(0, 0):,} | 📈 Pos: {test_pred_counts.get(1, 0):,}")
                    
                    st.divider()
                    
                    # ESEMPI DI PREDIZIONI
                    st.subheader("🔍 Esempi di Predizioni")
                    st.caption("Campione di recensioni classificate dal modello")
                    
                    # Formatta esempi per la visualizzazione
                    esempi_formatted = esempi.copy()
                    esempi_formatted['Label'] = esempi_formatted['label'].apply(lambda x: '✅ Pos' if x == 1 else '❌ Neg')
                    esempi_formatted['Predizione'] = esempi_formatted['prediction'].apply(lambda x: '✅ Pos' if x == 1 else '❌ Neg')
                    esempi_formatted['Confidence'] = esempi_formatted['probability'].apply(
                        lambda x: f"{max(x):.2%}" if isinstance(x, (list, tuple)) else "N/A"
                    )
                    esempi_formatted['Recensione (troncata)'] = esempi_formatted['review'].str[:100] + "..."
                    
                    # Mostra tabella
                    st.dataframe(
                        esempi_formatted[['Recensione (troncata)', 'Label', 'Predizione', 'Confidence']],
                        use_container_width=True,
                        hide_index=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Errore durante l'addestramento: {e}")
    else:
        st.info("Carica i dati prima di procedere.")

# PAGINA 3: MAPPA & CLUSTERING
elif page == "🗺️ Mappa & Clustering":
    st.header("🗺️ Mappa Geografica & Clustering Intelligente")
    st.markdown("""
    **Due analisi complementari:**
    - 📍 **Mappa**: Visualizzazione geografica degli hotel
    - 🎯 **Clustering**: Gruppi significativi basati su caratteristiche
    """)
    
    if st.session_state.df_hotel:
        # Selezione modalità
        mode = st.radio(
            "Seleziona visualizzazione:",
            ["📍 Mappa Geografica", "🎯 Clustering Intelligente"],
            horizontal=True
        )
        
        st.divider()
        
        # ========= SEZIONE A: MAPPA GEOGRAFICA =========
        if "Mappa" in mode:
            st.subheader("📍 Distribuzione Geografica Hotels")
            st.markdown("Visualizza tutti gli hotel sulla mappa, colorati in base al voto medio.")
            
            with st.expander("ℹ️ Come funziona"):
                st.markdown("""
                - Ogni punto rappresenta un hotel
                - Colori basati su voto medio (più alto = migliore)
                - Usa la mappa interattiva per esplorare zone geografiche
                """)
            
            # Opzioni filtro
            col1, col2 = st.columns(2)
            with col1:
                sample_size = st.slider("Percentuale hotel da mostrare", 10, 100, 50, step=10,
                                       help="Ridurre per performance migliori")
            with col2:
                min_reviews = st.number_input("Minimo recensioni", 0, 1000, 50,
                                             help="Filtra hotel con poche recensioni")
            
            if st.button("🗺️ Mostra Mappa", type="primary"):
                with st.spinner("Generando mappa..."):
                    try:
                        # Aggregazione: un punto per hotel
                        df_map_data = st.session_state.df_hotel.groupBy("Hotel_Name", "lat", "lng") \
                            .agg(
                                avg("Reviewer_Score").alias("voto_medio"),
                                count("*").alias("num_recensioni")
                            ) \
                            .filter(col("num_recensioni") >= min_reviews) \
                            .sample(fraction=sample_size/100.0) \
                            .toPandas()
                        
                        # Rinomina per Streamlit
                        df_map_data = df_map_data.rename(columns={'lng': 'longitude', 'lat': 'latitude'})
                        
                        # Metriche
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🏨 Hotels Visualizzati", len(df_map_data))
                        with col2:
                            avg_score = df_map_data['voto_medio'].mean()
                            st.metric("⭐ Voto Medio Globale", f"{avg_score:.2f}")
                        with col3:
                            total_reviews = df_map_data['num_recensioni'].sum()
                            st.metric("📝 Recensioni Totali", f"{total_reviews:,}")
                        
                        # Mappa
                        st.map(df_map_data[['latitude', 'longitude']], size=20)
                        
                        # Top 5 per voto
                        st.markdown("### 🏆 Top 5 Hotels per Voto")
                        top5 = df_map_data.nlargest(5, 'voto_medio')[['Hotel_Name', 'voto_medio', 'num_recensioni']]
                        top5.columns = ['Hotel', 'Voto Medio', 'Recensioni']
                        st.dataframe(top5, use_container_width=True, hide_index=True)
                        
                    except Exception as e:
                        st.error(f"Errore: {e}")
        
        # ========= SEZIONE B: CLUSTERING INTELLIGENTE =========
        else:
            st.subheader("🎯 Clustering Intelligente degli Hotel")
            
            # Introduzione semplificata
            st.markdown("""
            ### 💡 Cosa fa questa analisi?
            
            Invece di raggruppare hotel per **posizione geografica** (lat/lon), li raggruppa per **come sono davvero**:
            - Quanto sono buoni (voto, recensioni eccellenti)
            - Cosa dicono i clienti (sentiment positivo/negativo)
            - Quanti turisti diversi attraggono (nazionalità)
            - Quali problemi hanno (costruzioni, pulizia, staff)
            
            **Risultato**: Scopri gruppi come "Hotel di Lusso", "Tesori Nascosti", "Budget con Problemi"
            """)
            
            # Process visualization
            with st.expander("📋 Come funziona il processo (passo-passo)"):
                st.markdown("""
                #### Step 1️⃣: Calcolo Features per ogni Hotel
                Per ogni hotel, calcoliamo:
                - 📊 **Performance**: Voto medio, quante recensioni ha, % di voti eccellenti (≥9)
                - 😊 **Sentiment**: Quanto sono lunghe le recensioni positive vs negative
                - 🌍 **Diversità**: Quante nazionalità diverse lo recensiscono
                - ⚠️ **Problemi**: % di recensioni che menzionano costruzioni, sporco, staff scortese
                
                #### Step 2️⃣: Normalizzazione
                Trasformiamo tutti i numeri sulla stessa scala (0-1) così nessuna feature domina le altre.
                
                #### Step 3️⃣: K-Means Clustering
                L'algoritmo raggruppa hotel **simili** tra loro basandosi su tutte le features insieme.
                
                #### Step 4️⃣: Interpretazione Automatica
                Il sistema analizza ogni gruppo e suggerisce un nome (es. "Premium Hotels" se hanno voto alto e tante recensioni).
                """)
            
            st.divider()
            
            # Seleziona K con spiegazione
            st.markdown("### 🔢 Quanti gruppi vuoi trovare?")
            col1, col2 = st.columns([2, 1])
            with col1:
                k_clusters = st.slider("Numero di Gruppi (K)", 2, 6, 4)
            with col2:
                st.info(f"""
                **Consiglio:**
                - K=3: Pochi gruppi ben distinti
                - K=4: **Bilanciato** ✅
                - K=5-6: Molti gruppi dettagliati
                """)
            
            if st.button("🚀 Avvia Clustering Intelligente", type="primary", use_container_width=True):
                with st.spinner("⏳ Analisi in corso (può richiedere 30-60 secondi)..."):
                    
                    # Progress indicators
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        status_text.text("📊 Step 1/4: Calcolo features per ogni hotel...")
                        progress_bar.progress(25)
                        
                        result = gestore.esegui_clustering_hotel(st.session_state.df_hotel, k=k_clusters)
                        
                        status_text.text("🔢 Step 2/4: Normalizzazione e clustering...")
                        progress_bar.progress(50)
                        
                        df_clustered = result['df_clustered']
                        cluster_stats = result['cluster_stats'].toPandas()
                        cluster_names = result['cluster_names']
                        
                        status_text.text("📈 Step 3/4: Calcolo statistiche...")
                        progress_bar.progress(75)
                        
                        status_text.text("✅ Step 4/4: Generazione visualizzazioni...")
                        progress_bar.progress(100)
                        
                        # Clear progress
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.success(f"✅ **Completato!** Trovati {len(cluster_stats)} gruppi distinti di hotel.")
                        
                        # ===== LEGENDA =====
                        with st.expander("📖 LEGENDA - Come Leggere i Risultati", expanded=False):
                            st.markdown("""
                            ### 📊 Significato delle Metriche
                            
                            | Metrica | Cosa Significa | Esempio |
                            |---------|---------------|---------|
                            | **🏨 Hotels** | Quanti hotel ci sono in questo gruppo | 45 hotels |
                            | **⭐ Voto Medio** | Media dei voti di tutti gli hotel del gruppo | 8.5/10 |
                            | **📝 Recensioni Avg** | Media di quante recensioni ha ogni hotel | 1200 recensioni/hotel |
                            | **🌟 % Eccellenti** | % di recensioni con voto ≥ 9/10 (super soddisfatti) | 55% = metà clienti entusiasti |
                            | **⚠️ % Problemi** | % di recensioni che menzionano costruzioni/lavori | 12% = pochi problemi |
                            
                            ---
                            
                            ### 🏷️ Significato dei Nomi dei Gruppi
                            
                            I nomi sono **assegnati automaticamente** dal sistema in base alle caratteristiche:
                            
                            #### 🏆 Premium Hotels
                            - **Voto**: ≥ 8.5 (eccellente)
                            - **Recensioni**: > 500 (molto popolari)
                            - **Profilo**: Hotel di lusso consolidati, qualità garantita
                            
                            #### 💎 Hidden Gems (Tesori Nascosti)
                            - **Voto**: ≥ 8.0 (ottimo)
                            - **Recensioni**: < 200 (poca visibilità)
                            - **Profilo**: Piccoli hotel di qualità, poco conosciuti ma eccellenti
                            
                            #### 🌟 Popular Mixed
                            - **Voto**: 7.0-8.4 (medio-buono)
                            - **Recensioni**: > 800 (famosissimi)
                            - **Profilo**: Hotel molto noti ma con opinioni miste (alcuni adorano, altri no)
                            
                            #### 📉 Budget/Problems
                            - **Voto**: < 7.0 (basso) OR
                            - **Problemi**: > 15% (molte menzioni negative)
                            - **Profilo**: Hotel economici o con problemi ricorrenti
                            
                            #### 📊 Cluster X
                            - Gruppo che non rientra nelle categorie precedenti
                            - Guarda le metriche per capire il profilo
                            """)
                        
                        st.divider()
                        
                        # ===== SEZIONE RISULTATI SEMPLIFICATA =====
                        
                        st.markdown("## 🏷️ Gruppi Scoperti")
                        st.caption("Ogni gruppo rappresenta hotel con caratteristiche simili")
                        
                        st.markdown("")  # Spacing
                        
                        # Cards per ogni cluster
                        for idx, row in cluster_stats.iterrows():
                            cluster_id = int(row['cluster'])
                            nome = cluster_names.get(cluster_id, f"Gruppo {cluster_id}")
                            
                            # Visual container per cluster
                            st.markdown(f"### {nome}")
                            
                            # Metrics in colonne con spacing migliorato
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("🏨 Hotels", int(row['num_hotel']))
                            with col2:
                                st.metric("⭐ Voto Medio", f"{row['avg_voto']:.2f}")
                            with col3:
                                st.metric("📝 Recensioni Avg", f"{row['avg_recensioni']:.0f}")
                            with col4:
                                perc_ecc = row['avg_eccellenti']
                                st.metric("🌟 % Eccellenti", f"{perc_ecc:.1f}%")
                            
                            st.markdown("")  # Spacing
                            
                            # Interpretazione
                            if row['avg_voto'] >= 8.5:
                                st.success("✨ **Qualità Eccellente** - Hotel di alto livello con ottime recensioni")
                            elif row['avg_voto'] >= 7.5:
                                st.info("👍 **Buona Qualità** - Hotel solidi con feedback positivo")
                            else:
                                st.warning("⚠️ **Da Migliorare** - Possibili problemi da affrontare")
                            
                            st.markdown("")  # Spacing extra
                            st.divider()
                        
                        # Grafico comparativo semplificato
                        st.markdown("## 📊 Confronto Veloce")
                        
                        # Prepara dati per grafico
                        import pandas as pd
                        chart_data = pd.DataFrame({
                            'Gruppo': [cluster_names.get(int(r['cluster']), f"Gruppo {r['cluster']}") for _, r in cluster_stats.iterrows()],
                            'Voto Medio': cluster_stats['avg_voto'].values,
                            'Num Hotels': cluster_stats['num_hotel'].values
                        })
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("#### ⭐ Voto Medio per Gruppo")
                            st.bar_chart(chart_data.set_index('Gruppo')['Voto Medio'])
                        
                        with col2:
                            st.markdown("#### 🏨 Numero Hotel per Gruppo")
                            st.bar_chart(chart_data.set_index('Gruppo')['Num Hotels'])
                        
                        # Tabella dettagliata (collapsible)
                        with st.expander("📋 Vedi Statistiche Dettagliate"):
                            display_stats = cluster_stats.copy()
                            display_stats['Cluster'] = display_stats['cluster'].apply(lambda x: cluster_names.get(int(x), f"Gruppo {x}"))
                            display_stats = display_stats[['Cluster', 'num_hotel', 'avg_voto', 'avg_recensioni', 'avg_eccellenti', 'avg_problemi']]
                            display_stats.columns = ['Gruppo', '#Hotels', 'Voto', 'Recensioni', '% Eccellenti', '% Problemi']
                            
                            st.dataframe(display_stats, use_container_width=True, hide_index=True)
                        
                        # Esempi hotel (compatto)
                        with st.expander("🏆 Vedi Esempi di Hotel per Gruppo"):
                            df_examples = df_clustered.select("Hotel_Name", "cluster", "voto_medio", "num_recensioni") \
                                .orderBy("cluster", col("voto_medio").desc()) \
                                .limit(15) \
                                .toPandas()
                            
                            for cluster_id in sorted(df_examples['cluster'].unique()):
                                cluster_hotels = df_examples[df_examples['cluster'] == cluster_id].head(3)
                                nome_cluster = cluster_names.get(cluster_id, f"Gruppo {cluster_id}")
                                
                                st.markdown(f"**{nome_cluster}**")
                                for _, hotel in cluster_hotels.iterrows():
                                    st.caption(f"• {hotel['Hotel_Name']} - ⭐ {hotel['voto_medio']:.2f} ({hotel['num_recensioni']:.0f} rec)")
                                st.markdown("")
                        
                    except Exception as e:
                        st.error(f"❌ Errore durante il clustering: {e}")
                        with st.expander("🔍 Dettagli Tecnici"):
                            import traceback
                            st.code(traceback.format_exc())
    else:
        st.info("💡 Carica i dati dalla sidebar per iniziare.")

# PAGINA 4: TOPIC MODELING
elif page == "📝 Topic Modeling":
    st.header("📝 Topic Modeling (LDA)")
    st.markdown("""
    Scopri i **temi nascosti** nelle recensioni negative usando **Latent Dirichlet Allocation**. 
    LDA identifica automaticamente gruppi di parole che appaiono frequentemente insieme, rivelando i principali problemi lamentati dai clienti.
    """)
    
    # Spiegazione dettagliata
    with st.expander("ℹ️ Come funziona l'analisi?"):
        st.markdown("""
        **Processo di analisi:**
        1. 📋 **Selezione**: Uso solo recensioni negative con almeno 30 caratteri
        2. 🧹 **Pulizia**: Rimuovo punteggiatura, simboli (*, -, !, ecc.) e parole comuni ("the", "and", "hotel")
        3. 🔢 **Vettorizzazione**: Converto il testo in numeri analizzabili
        4. 🤖 **LDA**: Algoritmo che scopre quali parole tendono ad apparire insieme
        5. 📊 **Output**: Ogni "topic" è un gruppo di parole correlate che rappresentano un tema comune
        
        **Esempio interpretazione:**
        - Se vedi: `room`, `small`, `bed`, `bathroom` → Problema dimensioni camere
        - Se vedi: `breakfast`, `food`, `restaurant`, `menu` → Problema ristorazione
        - Se vedi: `staff`, `rude`, `service`, `reception` → Problema servizio
        
        **Nota**: Le parole sono ordinate per **peso** (numero tra parentesi) - più alto = più importante per quel topic.
        """)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        num_topics = st.slider("🔢 Numero di Topic da estrarre", 2, 10, 3, help="Più topic = più granularità, ma rischio di sovrapposizione")

        evaluate_stability = st.checkbox(
            "🧪 Valuta stabilità topic (2 run con seed diversi)",
            value=True,
            help="Calcola quanto i topic sono stabili tra due addestramenti LDA. Utile per validare interpretabilità."
        )

    with col2:
        st.metric("🎯 Consigliato", "3-4 topic")
    
    if "df_hotel" in st.session_state and st.session_state.df_hotel is not None:
        if st.button("🚀 Estrai Topic dalle Recensioni Negative", type="primary"):
            with st.spinner("🔍 Analisi LDA in corso..."):
                try:
                    result = gestore.esegui_topic_modeling(st.session_state.df_hotel, num_topics=num_topics, evaluate_stability=evaluate_stability, top_terms=10)

                    topics_data = result["topics_data"]
                    vocab = result["vocab"]
                    num_reviews = int(result["num_reviews"])
                    log_perplexity = float(result["log_perplexity"])
                    log_likelihood = float(result.get("log_likelihood", 0.0))

                    st.success(f"✅ Analisi completata su **{num_reviews:,}** recensioni negative!")

                    # Warning se dataset troppo piccolo per LDA stabile
                    if num_reviews < 300:
                        st.warning(
                            "⚠️ Campione relativamente piccolo per LDA: i topic potrebbero essere meno stabili. "
                            "Se possibile, riduci filtri (es. lunghezza minima) o usa meno topic."
                        )

                    # Metriche modello
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("📊 Recensioni analizzate", f"{num_reviews:,}")
                    with c2:
                        st.metric("📉 Log-Perplexity", f"{log_perplexity:.3f}",
                              help="Più basso tende a indicare un modello che si adatta meglio ai dati.")
                    with c3:
                        st.metric("📈 Log-Likelihood", f"{log_likelihood:.1f}",
                              help="Più alto (meno negativo) indica un fit migliore, a parità di dati e k.")

                    if evaluate_stability and result.get("stability") is not None:
                        st.markdown("### 🧪 Stabilità dei Topic")
                        stability = float(result["stability"])
                        seeds = result.get("compare_topics", {})
                        seed_main = seeds.get("seed_main", 42)
                        seed_alt = seeds.get("seed_alt", 99)

                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.metric("Stability score (Jaccard)", f"{stability:.3f}")
                        with c2:
                            st.caption(
                                f"Confronto tra due addestramenti LDA con seed {seed_main} e {seed_alt}. "
                                "0 = instabile, 1 = identico sui top-terms."
                            )

                        # Warning soglie pratiche
                        if stability < 0.35:
                            st.warning("⚠️ Stabilità bassa: i topic cambiano molto tra run. Prova a ridurre k, aumentare minDF o ripulire stopwords.")
                        elif stability < 0.55:
                            st.info("ℹ️ Stabilità media: topic ragionevoli ma con sovrapposizioni. Puoi migliorare con stopwords di dominio.")
                        else:
                            st.success("✅ Stabilità buona: i topic sono robusti e interpretabili.")

                        # Tabella per-topic
                        st.markdown("#### Dettaglio per Topic")
                        stab_df = pd.DataFrame(result["stability_table"])
                        stab_df["topic_main"] = stab_df["topic_main"].apply(lambda x: f"Topic {x+1}")
                        stab_df["topic_alt"] = stab_df["topic_alt"].apply(lambda x: f"Topic {x+1}")
                        stab_df["main_top_terms"] = stab_df["main_terms"].apply(lambda t: ", ".join(t[:7]))
                        stab_df["alt_top_terms"] = stab_df["alt_terms"].apply(lambda t: ", ".join(t[:7]))

                        st.dataframe(
                            stab_df[["topic_main", "topic_alt", "jaccard", "main_top_terms", "alt_top_terms"]],
                            use_container_width=True,
                            height=300
                        )

                    st.divider()
                    st.subheader("🗂️ Topic trovati")
                    st.caption("Ogni topic è una distribuzione di parole: qui mostriamo i termini più pesati.")

                    # Converti in pandas (pochi topic => sicuro)
                    topics_pdf = topics_data.toPandas().sort_values("topic")

                    for _, row in topics_pdf.iterrows():
                        topic_id = int(row["topic"])
                        term_indices = row["termIndices"]
                        term_weights = row["termWeights"]

                        # Mappa indici -> termini
                        terms_with_weights = [
                            (vocab[int(idx)], float(w))
                            for idx, w in zip(term_indices, term_weights)
                            if int(idx) < len(vocab)
                        ]

                        if len(terms_with_weights) == 0:
                            continue

                        # Normalizzazione robusta: rispetto al max peso del topic
                        max_w = max(w for _, w in terms_with_weights) or 1.0

                        st.markdown(f"### 📌 Topic {topic_id + 1}")

                        top_terms = terms_with_weights[:10]  # mostra 10 termini
                        for term, w in top_terms:
                            norm = w / max_w  # 0..1
                            st.progress(
                                float(norm),
                                text=f"**{term}**  (peso={w:.4f})"
                            )

                        # Interpretazione suggerita: prime 3 parole
                        st.caption(f"💡 **Etichetta suggerita**: {', '.join([t[0] for t in top_terms[:3]])}")
                    st.divider()

                except Exception as e:
                    st.error(f"❌ Errore durante l'analisi: {e}")
    else:
        st.info("💡 Carica i dati per iniziare l'analisi.")

# PAGINA 5: INSIGHT AVANZATI
elif page == "🧠 Insight Avanzati":
    st.header("🧠 Insight Avanzati")
    st.markdown("""
    Analisi avanzate con query Spark personalizzate per scoprire pattern nascosti nei dati.
    """)
    
    if st.session_state.df_hotel:
        # === SELEZIONE QUERY NEL SIDEBAR ===
        st.sidebar.divider()
        st.sidebar.header("🔍 Configurazione Query")
        
        query_type = st.sidebar.selectbox(
            "Tipo di analisi:",
            ["🌍 Nazionalità", "🏗️ Lavori in Corso", "👥 Tipo Viaggio", "📏 Lunghezza Recensioni", "📉 Affidabilità Voto (Std Dev)", "⚠️ Hotel Rischiosi (Alto Rischio)", "🤯 Expectation Gap (Realtà vs Aspettativa)"],
            help="Seleziona il tipo di analisi avanzata da eseguire"
        )
        
        st.markdown("")  # Spacing
        st.divider()
        
        # ========= QUERY 1: NAZIONALITÀ =========
        if "Nazionalità" in query_type:
            st.subheader("🌍 Analisi per Nazionalità (stile di valutazione)")
            st.markdown("""
            **Obiettivo**: osservare differenze nello **stile di valutazione** (più severo vs più indulgente) tra gruppi di recensori.  
            **Nota**: non misura “qualità degli hotel” ma **tendenza media del voto** per nazionalità, su campioni numericamente significativi.
            """)

            with st.expander("ℹ️ Come funziona"):
                st.markdown("""
                - **Filtro**: solo nazionalità con >100 recensioni (maggiore affidabilità)
                - **Metriche**: voto medio, deviazione standard (σ), min/max
                - **Interpretazione**:
                - **Voto medio più basso** → valutazione mediamente più severa
                - **σ alta** → giudizi più discordanti (love/hate)
                """)

            if st.button("🚀 Esegui Analisi Nazionalità", type="primary"):
                with st.spinner("Analisi in corso..."):
                    df_naz = gestore.query_nazionalita_critiche(st.session_state.df_hotel).toPandas()

                if df_naz is None or len(df_naz) == 0:
                    st.warning("Nessun dato trovato per questa analisi.")
                else:
                    # Assicuriamoci dei nomi colonna corretti
                    # Backend nuovo: nationality_clean
                    if "nationality_clean" not in df_naz.columns and "Reviewer_Nationality" in df_naz.columns:
                        df_naz = df_naz.rename(columns={"Reviewer_Nationality": "nationality_clean"})

                    # Metriche generali
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("📌 Nazionalità analizzate", f"{len(df_naz)}")
                    with c2:
                        std_media = float(df_naz["deviazione_std"].mean())
                        st.metric("📊 σ media", f"{std_media:.2f}")
                    with c3:
                        range_voti = float(df_naz["voto_medio"].max() - df_naz["voto_medio"].min())
                        st.metric("📈 Range voto medio", f"{range_voti:.2f}")

                    st.divider()

                    # Top/Bottom 5 robusti
                    left, right = st.columns(2)

                    with left:
                        st.markdown("### 😊 Top 5 - Più Generosi")
                        # FIX: Usa 'nationality_clean' invece di 'Reviewer_Nationality' per match con backend
                        top5 = df_naz.nlargest(5, 'voto_medio')[['nationality_clean', 'voto_medio', 'num_recensioni']]
                        top5.columns = ['Nazionalità', 'Voto Medio', 'Recensioni']
                        st.dataframe(top5, use_container_width=True, hide_index=True)

                    with right:
                        st.markdown("### � Top 5 - Più Critici")
                        # FIX: Usa 'nationality_clean' invece di 'Reviewer_Nationality' per match con backend
                        bottom5 = df_naz.nsmallest(5, 'voto_medio')[['nationality_clean', 'voto_medio', 'num_recensioni']]
                        bottom5.columns = ['Nazionalità', 'Voto Medio', 'Recensioni']
                        st.dataframe(bottom5, use_container_width=True, hide_index=True)

                    st.divider()

                    # Grafico: top15 + bottom15 espliciti
                    st.markdown("### 📊 Confronto visivo (15 più severi + 15 più indulgenti)")
                    import pandas as pd

                    bottom15 = df_naz.nsmallest(15, "voto_medio")
                    top15 = df_naz.nlargest(15, "voto_medio")

                    df_viz = pd.concat([bottom15, top15], axis=0)
                    df_viz["label"] = df_viz["nationality_clean"] + " (" + df_viz["num_recensioni"].astype(int).astype(str) + ")"

                    # Ordina per voto medio così il grafico è leggibile
                    df_viz = df_viz.sort_values("voto_medio", ascending=True)

                    st.bar_chart(df_viz.set_index("label")["voto_medio"])

                    st.caption("Etichetta = Nazionalità (numero recensioni). σ e min/max sono disponibili nelle tabelle sopra.")
        
        # ========= QUERY 2: COSTRUZIONI =========
        elif "Lavori" in query_type:
            st.subheader("🏗️ Impatto dei Lavori in Corso")
            st.markdown("""
            **Obiettivo**: stimare quanto le menzioni di ristrutturazioni/costruzioni impattino sul voto.  
            **Metodo**: confronto tra gruppi (con lavori vs senza) + keyword più frequenti + campioni di recensioni.
            """)

            with st.expander("ℹ️ Come funziona"):
                st.markdown("""
                - **Pattern lavori**: construction / renovation / drilling / hammering / works / building work  
                - **Output**:
                1) Statistiche per gruppo (media, σ, CI95)  
                2) Top keyword principali nelle recensioni con lavori  
                3) Esempi di recensioni reali (campioni)
                """)

            with st.expander("⚙️ Impostazioni", expanded=False):
                sample_size = st.slider("Numero esempi recensioni", 3, 15, 5)

            if st.button("🚀 Esegui Analisi Lavori", type="primary"):
                with st.spinner("Cercando recensioni con menzioni di lavori..."):
                    # FIX: Il backend restituisce un dict, estraiamo 'stats_df' prima di convertire
                    result = gestore.query_impatto_costruzioni(st.session_state.df_hotel, sample_size=sample_size)

                # --- Compatibilità: se backend vecchio ritorna DF, adattiamo ---
                if isinstance(result, dict):
                    stats_pdf = result["stats_df"].toPandas()
                    kw_pdf = result["keywords_df"].toPandas()
                    samples_pdf = result["samples_df"].toPandas()
                else:
                    # backend vecchio: solo stats
                    stats_pdf = result.toPandas()
                    kw_pdf = None
                    samples_pdf = None

                if stats_pdf is None or len(stats_pdf) == 0:
                    st.warning("Dati insufficienti per l'analisi.")
                else:
                    # Label leggibili
                    stats_pdf["gruppo"] = stats_pdf["has_construction"].map({True: "🏗️ Con lavori", False: "🏨 Senza lavori"})

                    # Estrai gruppi se presenti
                    row_no = stats_pdf[stats_pdf["has_construction"] == False]
                    row_yes = stats_pdf[stats_pdf["has_construction"] == True]

                    voto_no = float(row_no["voto_medio"].values[0]) if len(row_no) else None
                    voto_yes = float(row_yes["voto_medio"].values[0]) if len(row_yes) else None
                    n_yes = int(row_yes["totale"].values[0]) if len(row_yes) else 0

                    ci_no = float(row_no["ci95"].values[0]) if (len(row_no) and "ci95" in row_no.columns) else None
                    ci_yes = float(row_yes["ci95"].values[0]) if (len(row_yes) and "ci95" in row_yes.columns) else None

                    # KPI
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("🏨 Senza lavori (media)", f"{voto_no:.2f}" if voto_no is not None else "n/a",
                                  help=f"CI95 ± {ci_no:.2f}" if ci_no is not None else None)
                    with c2:
                        if voto_no is not None and voto_yes is not None:
                            diff = voto_yes - voto_no
                            st.metric("🏗️ Con lavori (media)", f"{voto_yes:.2f}",
                                      delta=f"{diff:.2f}" if diff < 0 else f"+{diff:.2f}",
                                      delta_color="inverse" if diff < 0 else "normal",
                                      help=f"CI95 ± {ci_yes:.2f}" if ci_yes is not None else None)
                        else:
                            st.metric("🏗️ Con lavori (media)", "n/a")
                    with c3:
                        st.metric("📊 Recensioni con lavori", f"{n_yes:,}")

                    # Interpretazione “più corretta”
                    if voto_no is not None and voto_yes is not None:
                        diff = voto_yes - voto_no

                        # Se abbiamo CI, confrontiamo in modo conservativo
                        if ci_no is not None and ci_yes is not None:
                            # se |diff| > somma delle incertezze, è un segnale forte (euristica conservativa)
                            threshold = ci_no + ci_yes
                            if diff < -threshold:
                                st.error(f"⚠️ **Impatto forte**: differenza **{diff:.2f}** oltre l'incertezza (±{threshold:.2f}).")
                            elif diff < 0:
                                st.warning(f"🔸 Impatto negativo ma vicino all'incertezza: differenza **{diff:.2f}** (±{threshold:.2f}).")
                            else:
                                st.success("✅ Nessun impatto negativo netto rilevato.")
                        else:
                            # fallback: soglie euristiche
                            if diff < -0.3:
                                st.error(f"⚠️ Impatto significativo: i lavori riducono il voto di **{abs(diff):.2f}** punti.")
                            elif diff < 0:
                                st.warning(f"🔸 Impatto moderato: riduzione di **{abs(diff):.2f}** punti.")
                            else:
                                st.success("✅ Nessun impatto negativo rilevato.")

                st.divider()
                # Grafico: media con (eventuali) error bars CI95
                st.markdown("### 📈 Confronto visivo")
                st.caption("Barre = voto medio. Se presente, barre d’errore = CI95.")

                import altair as alt

                base = alt.Chart(stats_pdf).encode(
                    x=alt.X("gruppo:N", title="Gruppo"),
                    tooltip=[
                        alt.Tooltip("gruppo:N", title="Gruppo"),
                        alt.Tooltip("voto_medio:Q", title="Voto medio", format=".2f"),
                        alt.Tooltip("totale:Q", title="# recensioni"),
                        alt.Tooltip("deviazione_std:Q", title="σ", format=".2f") if "deviazione_std" in stats_pdf.columns else alt.value(None),
                        alt.Tooltip("ci95:Q", title="CI95", format=".2f") if "ci95" in stats_pdf.columns else alt.value(None),
                    ]
                )

                bars = base.mark_bar().encode(
                    y=alt.Y("voto_medio:Q", title="Voto medio")
                )

                if "ci95" in stats_pdf.columns and stats_pdf["ci95"].notna().any():
                    err = base.mark_errorbar().encode(
                        y=alt.Y("voto_medio:Q"),
                        yError=alt.YError("ci95:Q")
                    )
                    st.altair_chart((bars + err).properties(height=320), use_container_width=True)
                else:
                    st.altair_chart(bars.properties(height=320), use_container_width=True)

                # Keyword freq
                if kw_pdf is not None and len(kw_pdf) > 0:
                    st.divider()
                    st.markdown("### 🔑 Keyword più frequenti (recensioni con lavori)")
                    st.caption("Serve a capire quale aspetto dei lavori è più citato (drilling vs renovation vs construction).")

                    top_kw = kw_pdf.head(10)
                    st.dataframe(top_kw, use_container_width=True, height=280)

                # Samples
                if samples_pdf is not None and len(samples_pdf) > 0:
                    st.divider()
                    st.markdown("### 🧾 Esempi reali di recensioni (campione)")
                    for i, r in samples_pdf.iterrows():
                        with st.expander(f"{r['Hotel_Name']} | score={r['Reviewer_Score']} | kw={r.get('kw_main','')}", expanded=False):
                            st.write(r["Negative_Review"])
        
        # ========= QUERY 3: TIPO VIAGGIO =========
        elif "Tipo Viaggio" in query_type:
            st.subheader("👥 Analisi per Tipo di Viaggio")
            st.markdown("""
            **Obiettivo**: capire quale target (coppie, famiglie, solitari, gruppi) è più soddisfatto.  
            **Utilità**: supportare scelte di marketing/servizi in base al segmento più “critico” o più “profittevole”.
            (Nota: qui misuriamo soddisfazione tramite `Reviewer_Score` e consistenza tramite dispersione.)
            """)

            with st.expander("ℹ️ Come funziona"):
                st.markdown("""
                - **Categorie**: Coppia, Famiglia, Solo, Gruppo, Altro  
                - **Filtro**: solo gruppi con > 50 recensioni  
                - **Interpretazione**:  
                - **Voto medio alto** ⇒ target più soddisfatto  
                - **σ bassa** ⇒ esperienza più consistente  
                - **CI95 (se presente)** ⇒ incertezza della media (più piccolo = stima più affidabile)
                """)

            with st.expander("⚙️ Impostazioni", expanded=False):
                show_table = st.checkbox("Mostra tabella completa", value=True)

            if st.button("🚀 Esegui Analisi Tipo Viaggio", type="primary"):
                with st.spinner("Estraendo tag di viaggio..."):
                    df_viaggi = gestore.query_coppie_vs_famiglie(st.session_state.df_hotel).toPandas()

                if df_viaggi is None or len(df_viaggi) == 0:
                    st.warning("Nessun dato trovato.")
                else:
                    # Arrotonda per UI (evita style)
                    for c in ["voto_medio", "deviazione_std", "ci95"]:
                        if c in df_viaggi.columns:
                            df_viaggi[c] = df_viaggi[c].astype(float).round(3)

                    # Best target robusto (non dipende dall'ordine)
                    best = df_viaggi.loc[df_viaggi["voto_medio"].idxmax()]
                    worst = df_viaggi.loc[df_viaggi["voto_medio"].idxmin()]

                    # KPI cards (layout fisso)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("🏆 Target più soddisfatto", best["tipo_viaggio"])
                    with c2:
                        st.metric("⭐ Voto medio (best)", f"{best['voto_medio']:.2f}/10")
                    with c3:
                        st.metric("😬 Target meno soddisfatto", worst["tipo_viaggio"])

                    st.divider()

                    st.markdown("### 📈 Confronto visivo (media + affidabilità)")
                    st.caption("Barre = voto medio. Se presente, barre d’errore = CI95 (1.96 * std/sqrt(n)). Tooltip con σ e #recensioni.")

                    import altair as alt

                    # grafico base: bar del voto medio
                    base = alt.Chart(df_viaggi).encode(
                        x=alt.X("tipo_viaggio:N", sort="-y", title="Tipo viaggio"),
                        tooltip=[
                            alt.Tooltip("tipo_viaggio:N", title="Categoria"),
                            alt.Tooltip("voto_medio:Q", title="Voto medio", format=".2f"),
                            alt.Tooltip("num_recensioni:Q", title="# recensioni"),
                            alt.Tooltip("deviazione_std:Q", title="σ", format=".2f"),
                        ],
                    )

                    bars = base.mark_bar().encode(
                        y=alt.Y("voto_medio:Q", title="Voto medio")
                    )

                    if "ci95" in df_viaggi.columns and df_viaggi["ci95"].notna().any():
                        # Error bars: mean ± ci95
                        err = base.mark_errorbar().encode(
                            y=alt.Y("voto_medio:Q"),
                            yError=alt.YError("ci95:Q")
                        )
                        chart = (bars + err).properties(height=380)
                    else:
                        chart = bars.properties(height=380)

                    st.altair_chart(chart, use_container_width=True)

                    st.divider()

                    st.markdown("### 🧠 Insight")
                    msg = (
                        f"Il target **più soddisfatto** risulta **{best['tipo_viaggio']}** "
                        f"con voto medio **{best['voto_medio']:.2f}**."
                    )
                    if "ci95" in df_viaggi.columns and not pd.isna(best.get("ci95", None)):
                        msg += f" (CI95 ± {best['ci95']:.2f})"
                    st.success("🏆 " + msg)

                    # Tabella completa
                    if show_table:
                        st.markdown("### 📋 Dettaglio completo")
                        cols = ["tipo_viaggio", "voto_medio", "num_recensioni", "deviazione_std"]
                        if "ci95" in df_viaggi.columns:
                            cols.append("ci95")
                        st.dataframe(df_viaggi[cols], use_container_width=True, height=320)

        # ========= QUERY 4: ASIMMETRIA EMOTIVA (Lunghezza) =========
        elif query_type == "📏 Lunghezza Recensioni":
            st.subheader("📏 Asimmetria Emotiva: la delusione genera più testo?")
            st.markdown("""
            **Obiettivo**: misurare l'asimmetria emotiva in modo quantitativo.  
            Usiamo le word-count già presenti nel dataset (parte positiva vs negativa).  
            - **Delta** = (negativo positivo) ⇒ quanto “sfogo” in più c'è nella parte negativa  
            - **Negativity ratio** = negativo / positivo ⇒ metrica relativa (solo se il positivo non è troppo piccolo)  
            - **% presenza testo** ⇒ quanto spesso le persone scrivono davvero una parte positiva/negativa
            """)

            with st.expander("⚙️ Impostazioni", expanded=False):
                show_table = st.checkbox("Mostra tabella completa", value=True)

            if st.button("🚀 Analizza Comportamento Emotivo", type="primary"):
                with st.spinner("Calcolando asimmetria emotiva..."):
                    df_emo = gestore.query_lunghezza_recensioni(st.session_state.df_hotel).toPandas()

                if df_emo is None or len(df_emo) == 0:
                    st.warning("Nessun dato trovato.")
                else:
                    # --- Ordine logico bucket ---
                    ORDER = [
                        "😠 < 5.0 (Arrabbiato)",
                        "😐 5.0-7.5 (Deluso)",
                        "🙂 7.5-9.0 (Soddisfatto)",
                        "😍 > 9.0 (Felice)"
                    ]
                    df_emo["score_bucket"] = df_emo["score_bucket"].astype(str)
                    df_emo["bucket_order"] = df_emo["score_bucket"].apply(lambda x: ORDER.index(x) if x in ORDER else 999)
                    df_emo = df_emo.sort_values("bucket_order").drop(columns=["bucket_order"])

                    # --- Compatibilità: se il backend non ha le nuove colonne, creale ---
                    if "delta_len_neg_minus_pos" not in df_emo.columns:
                        df_emo["delta_len_neg_minus_pos"] = df_emo["avg_negative_length"] - df_emo["avg_positive_length"]
                    if "pct_has_negative" not in df_emo.columns:
                        df_emo["pct_has_negative"] = None
                    if "pct_has_positive" not in df_emo.columns:
                        df_emo["pct_has_positive"] = None

                    st.markdown("### 📊 Risultati per fascia di voto")

                    cols = st.columns(min(4, len(df_emo)))
                    for i, row in enumerate(df_emo.itertuples(index=False)):
                        bucket = getattr(row, "score_bucket")
                        neg_len = float(getattr(row, "avg_negative_length"))
                        pos_len = float(getattr(row, "avg_positive_length"))
                        delta = float(getattr(row, "delta_len_neg_minus_pos"))

                        ratio = getattr(row, "negativity_ratio", None)
                        ratio_str = "n/a" if ratio is None else f"{float(ratio):.2f}x"

                        with cols[i % len(cols)]:
                            # metrica principale: delta
                            st.metric(
                                label=bucket,
                                value=f"{delta:.1f}",
                                delta="Δ (neg - pos) parole",
                                delta_color="inverse" if delta > 0 else "normal",
                                help=f"Neg: {neg_len:.0f} parole | Pos: {pos_len:.0f} parole | Ratio: {ratio_str}"
                            )

                            # secondaria: ratio 
                            st.caption(f"Ratio (neg/pos): **{ratio_str}**")

                            # presenza testo 
                            pn = getattr(row, "pct_has_negative", None)
                            pp = getattr(row, "pct_has_positive", None)
                            if pn is not None and pp is not None:
                                st.caption(f"Testo Neg presente: **{float(pn):.1f}%** | Pos presente: **{float(pp):.1f}%**")

                    st.divider()

                    # --- Grafico con Altair  ---
                    st.markdown("### 📉 Positive vs Negative: lunghezza media per fascia")
                    st.caption("Confronto tra lunghezza media della parte positiva e negativa (parole).")

                    import altair as alt
                    import pandas as pd

                    chart_data = pd.melt(
                        df_emo,
                        id_vars=["score_bucket"],
                        value_vars=["avg_positive_length", "avg_negative_length"],
                        var_name="Tipo",
                        value_name="Lunghezza_media"
                    )
                    chart_data["Tipo"] = chart_data["Tipo"].replace({
                        "avg_positive_length": "Positivo",
                        "avg_negative_length": "Negativo"
                    })

                    chart = (
                        alt.Chart(chart_data)
                        .mark_bar()
                        .encode(
                            x=alt.X("score_bucket:N", sort=ORDER, title="Fascia voto"),
                            y=alt.Y("Lunghezza_media:Q", title="Lunghezza media (parole)"),
                            color=alt.Color("Tipo:N", title="Tipo"),
                            xOffset="Tipo:N",
                            tooltip=[
                                alt.Tooltip("score_bucket:N", title="Fascia"),
                                alt.Tooltip("Tipo:N", title="Tipo"),
                                alt.Tooltip("Lunghezza_media:Q", title="Parole medie", format=".1f")
                            ]
                        )
                        .properties(height=380)
                    )

                    st.altair_chart(chart, use_container_width=True)

                    st.divider()

                    # --- Insight ---
                    st.markdown("### 🧠 Insight")

                    # Fascia con delta più alto = più sfogo negativo rispetto al positivo
                    idx_max = df_emo["delta_len_neg_minus_pos"].astype(float).idxmax()
                    idx_min = df_emo["delta_len_neg_minus_pos"].astype(float).idxmin()

                    max_row = df_emo.loc[idx_max]
                    min_row = df_emo.loc[idx_min]

                    if float(max_row["delta_len_neg_minus_pos"]) > 10:
                        st.warning(
                            f"⚠️ **Effetto sfogo**: nella fascia **{max_row['score_bucket']}** "
                            f"la parte negativa è mediamente più lunga di **{max_row['delta_len_neg_minus_pos']:.1f} parole** "
                            f"rispetto alla positiva."
                        )
                    else:
                        st.info("ℹ️ Nessuna asimmetria forte: le lunghezze positive/negative sono relativamente bilanciate.")

                    st.caption(
                        f"Fascia più 'negativity-heavy': {max_row['score_bucket']} | "
                        f"fascia più 'positivity-heavy': {min_row['score_bucket']}"
                    )

                    if show_table:
                        st.markdown("### 📋 Tabella completa")
                        st.dataframe(df_emo, use_container_width=True, height=420)

        # ========= QUERY 5: AFFIDABILITÀ VOTO =========
        elif "Affidabilità Voto" in query_type:
            st.subheader("📉 Affidabilità del Voto (Coerenza)")
            st.markdown("""
            **Obiettivo**: stimare quanto il punteggio medio di un hotel sia “affidabile” misurando la dispersione dei voti.  
            **Metrica**: **deviazione standard (σ)** dei `Reviewer_Score`.
            - **σ alta** ⇒ hotel polarizzante (esperienze molto diverse)
            - **σ bassa** ⇒ hotel consistente (esperienza prevedibile)
            """)

            with st.expander("⚙️ Impostazioni", expanded=False):
                min_reviews = st.number_input("Min recensioni per hotel", 0, 1000000, 100, step=50)
                top_k = st.slider("Top hotel da mostrare", 3, 50, 10)

            if st.button("🚀 Analizza Affidabilità", type="primary"):
                with st.spinner("Calcolando dispersione voti..."):
                    df_std = gestore.query_affidabilita_voto(st.session_state.df_hotel).toPandas()

                if df_std is None or len(df_std) == 0:
                    st.warning("Nessun dato sufficiente per l'analisi.")
                else:
                    # Compatibilità colonne (backend vecchio/nuovo)
                    if "avg_hotel_score" not in df_std.columns and "Average_Score" in df_std.columns:
                        df_std["avg_hotel_score"] = df_std["Average_Score"]

                    # Filtro UI
                    if "num_reviews" in df_std.columns:
                        df_std = df_std[df_std["num_reviews"] >= min_reviews].copy()

                    if len(df_std) == 0:
                        st.warning("Dopo il filtro, non restano hotel con abbastanza recensioni.")
                    else:
                        # Ordina per più controversi
                        df_std = df_std.sort_values("stddev_reviewer_score", ascending=False)

                        st.markdown("### 🔥 Hotel più controversi (σ alta)")
                        top = df_std.head(max(3, min(top_k, len(df_std))))

                        cols = st.columns(min(3, len(top)))
                        for i in range(min(3, len(top))):
                            row = top.iloc[i]
                            with cols[i]:
                                st.error(f"**{row['Hotel_Name']}**")
                                st.metric("Dispersione (σ)", f"{row['stddev_reviewer_score']:.2f}",
                                        delta="Polarizzante", delta_color="inverse")
                                st.caption(
                                    f"Media reviewer: {row['mean_reviewer_score']:.2f} | "
                                    f"Media hotel: {row['avg_hotel_score']:.2f} | "
                                    f"Recensioni: {int(row['num_reviews'])}"
                                )

                        st.divider()

                        st.markdown("### 📈 Mappa: Qualità vs Affidabilità")
                        st.caption("X = media voti, Y = deviazione standard. Dimensione = numero recensioni. Tooltip con dettagli.")

                        import altair as alt

                        chart = (
                            alt.Chart(df_std)
                            .mark_circle(opacity=0.85)
                            .encode(
                                x=alt.X("mean_reviewer_score:Q", title="Media Reviewer Score"),
                                y=alt.Y("stddev_reviewer_score:Q", title="Deviazione standard σ"),
                                size=alt.Size("num_reviews:Q", title="# recensioni", scale=alt.Scale(zero=False)),
                                color=alt.Color("stddev_reviewer_score:Q", title="σ (dispersione)"),
                                tooltip=[
                                    alt.Tooltip("Hotel_Name:N", title="Hotel"),
                                    alt.Tooltip("avg_hotel_score:Q", title="Media hotel", format=".2f"),
                                    alt.Tooltip("mean_reviewer_score:Q", title="Media reviewer", format=".2f"),
                                    alt.Tooltip("stddev_reviewer_score:Q", title="σ", format=".2f"),
                                    alt.Tooltip("num_reviews:Q", title="# recensioni"),
                                ],
                            )
                            .properties(height=480)
                        )
                        st.altair_chart(chart, use_container_width=True)

                        st.info(
                            "💡 **Lettura**: punti più in alto ⇒ maggiore disaccordo tra clienti (hotel imprevedibile). "
                            "Punti a destra e in basso ⇒ qualità alta e consistente."
                        )

                        st.markdown("### 📋 Dettaglio completo")
                        # Evito .style: più stabile
                        st.dataframe(df_std.head(top_k), use_container_width=True, height=420)

        # ========= QUERY 6: HOTEL RISCHIOSI =========
        elif "Hotel Rischiosi" in query_type:
            st.subheader("⚠️ Hotel 'Rischiosi' (Alta Media, Alto Rischio)")
            st.markdown("""
            **Obiettivo**: individuare hotel che sembrano eccellenti (media alta) ma nascondono una quota preoccupante di disastri (voti ≤ 4.0).  
            **Perché è utile?**: la media può “nascondere” una coda di esperienze pessime (es. pulizia, rumore, sicurezza, staff).
            """)

            with st.expander("⚙️ Impostazioni", expanded=False):
                min_reviews_ui = st.number_input("Min recensioni per hotel", 0, 1000000, 50, step=10)
                min_avg_ui = st.slider("Soglia media (apparenza ottima)", 0.0, 10.0, 8.0, 0.1)
                min_disaster_pct_ui = st.slider("Soglia % disastri (≤ 4.0)", 0.0, 50.0, 5.0, 0.5)
                show_table_ui = st.checkbox("Mostra tabella completa", value=True)

            if st.button("🚀 Scansiona Rischi Nascosti", type="primary"):
                with st.spinner("Cercando hotel rischiosi..."):
                    df_risky = gestore.query_hotel_rischiosi(st.session_state.df_hotel).toPandas()

                if df_risky is None or len(df_risky) == 0:
                    st.success("✅ Nessun hotel rischioso trovato con i criteri attuali.")
                else:
                    # ---- Normalizzazione nomi colonne (compatibilità backend vecchio/nuovo) ----
                    # Backend nuovo: avg_hotel_score, risk_index, p05_score
                    # Backend vecchio: Average_Score
                    if "avg_hotel_score" not in df_risky.columns and "Average_Score" in df_risky.columns:
                        df_risky["avg_hotel_score"] = df_risky["Average_Score"]

                    if "risk_index" not in df_risky.columns:
                        # fallback: ranking = disaster_pct
                        df_risky["risk_index"] = df_risky["disaster_pct"]

                    if "p05_score" not in df_risky.columns:
                        df_risky["p05_score"] = None  # opzionale

                    # ---- Filtri UI (nel caso backend abbia soglie diverse) ----
                    df_risky = df_risky[
                        (df_risky["total_reviews"] >= min_reviews_ui) &
                        (df_risky["avg_hotel_score"] >= min_avg_ui) &
                        (df_risky["disaster_pct"] >= min_disaster_pct_ui)
                    ].copy()

                    if len(df_risky) == 0:
                        st.success("✅ Nessun hotel rischioso trovato con i criteri attuali.")
                    else:
                        # Ordina per rischio
                        df_risky = df_risky.sort_values(["risk_index", "disaster_pct"], ascending=[False, False])

                        st.error(f"⚠️ Trovati **{len(df_risky)}** hotel 'ottimi' ma con rischio elevato!")

                        # ---- Top 3 ----
                        st.markdown("### 🔥 Top 3 potenziali 'trappole'")
                        cols = st.columns(min(3, len(df_risky)))
                        top_n = min(3, len(df_risky))

                        for i in range(top_n):
                            row = df_risky.iloc[i]
                            with cols[i]:
                                st.error(f"**{row['Hotel_Name']}**")
                                st.metric(
                                    "Disastri (≤ 4.0)",
                                    f"{row['disaster_pct']:.2f}%",
                                    delta="Rischio alto",
                                    delta_color="inverse"
                                )
                                st.caption(
                                    f"⭐ Avg hotel: {row['avg_hotel_score']:.2f} | "
                                    f"⭐ Avg reviewer: {row['mean_reviewer_score']:.2f} | "
                                    f"🧾 Review: {int(row['total_reviews'])} | "
                                    f"💥 Disastri: {int(row['disaster_count'])}"
                                )
                                if pd.notna(row.get("p05_score", None)):
                                    st.caption(f"📉 P05 score: {row['p05_score']:.2f}")

                        st.divider()

                        # ---- Scatter robusto con Altair (color + size funzionano sempre) ----
                        st.markdown("### 📉 Analisi visiva: ottimi ma pericolosi")
                        st.caption("Asse X: media dei reviewer. Asse Y: % disastri (≤ 4.0). Dimensione: # recensioni. Colore: risk_index.")

                        import altair as alt

                        chart = (
                            alt.Chart(df_risky)
                            .mark_circle(opacity=0.85)
                            .encode(
                                x=alt.X("mean_reviewer_score:Q", title="Media Reviewer Score"),
                                y=alt.Y("disaster_pct:Q", title="% Disastri (≤ 4.0)"),
                                size=alt.Size("total_reviews:Q", title="Totale recensioni", scale=alt.Scale(zero=False)),
                                color=alt.Color("risk_index:Q", title="Risk Index"),
                                tooltip=[
                                    alt.Tooltip("Hotel_Name:N", title="Hotel"),
                                    alt.Tooltip("avg_hotel_score:Q", title="Avg hotel", format=".2f"),
                                    alt.Tooltip("mean_reviewer_score:Q", title="Avg reviewer", format=".2f"),
                                    alt.Tooltip("disaster_pct:Q", title="% disastri", format=".2f"),
                                    alt.Tooltip("disaster_count:Q", title="# disastri"),
                                    alt.Tooltip("total_reviews:Q", title="# recensioni"),
                                    alt.Tooltip("risk_index:Q", title="risk_index", format=".2f"),
                                ],
                            )
                            .properties(height=460)
                        )
                        st.altair_chart(chart, use_container_width=True)

                        st.info(
                            "💡 **Lettura**: più in alto ⇒ maggiore probabilità di un’esperienza pessima. "
                            "Il colore e la dimensione aiutano a distinguere rischi 'affidabili' (molte recensioni) da outlier."
                        )

                        # ---- Tabella completa ----
                        if show_table_ui:
                            st.markdown("### 📋 Lista completa")
                            show_cols = [
                                "Hotel_Name", "avg_hotel_score", "mean_reviewer_score",
                                "disaster_pct", "disaster_count", "total_reviews", "risk_index", "p05_score"
                            ]
                            show_cols = [c for c in show_cols if c in df_risky.columns]

                            df_show = df_risky[show_cols].copy()
                            # Formattazione semplice (senza .style per evitare problemi)
                            st.dataframe(df_show, use_container_width=True, height=420)

        # ========= QUERY 7: EXPECTATION GAP =========
        elif "Expectation Gap" in query_type:
            st.subheader("🤯 Expectation Gap (Realtà vs Aspettativa)")

            st.markdown("""
            **Obiettivo**: misurare la delusione *relativa* (scarto tra voto dato e aspettativa media dell’hotel).  
            **Definizione**: `gap = Reviewer_Score − Average_Score`  
            - gap > 0 ⇒ esperienza migliore delle aspettative  
            - gap < 0 ⇒ esperienza peggiore delle aspettative
            """)

            with st.expander("ℹ️ Guida alla lettura"):
                st.markdown("""
                - **Gap medio**: indica se, in media, una fascia supera o disattende le aspettative.
                - **% delusioni**: quante recensioni hanno gap < 0 (fallimento rispetto alle aspettative).
                - **Intensità della delusione**: media del gap *solo* sulle recensioni negative (gap < 0).  
                Più è negativo ⇒ più “profonda” è la delusione quando succede.
                """)

            # Parametri (optional, ma utili)
            with st.expander("⚙️ Impostazioni analisi", expanded=False):
                min_reviews = st.number_input("Min recensioni per fascia (solo per stabilità statistica)", 0, 1000000, 0, step=100)
                show_table = st.checkbox("Mostra tabella dati", value=True)

            if st.button("🚀 Analizza il Gap", type="primary"):
                with st.spinner("Calcolo in corso..."):
                    df_gap = gestore.query_expectation_gap(st.session_state.df_hotel).toPandas()

                if df_gap is None or len(df_gap) == 0:
                    st.warning("Nessun dato trovato.")
                else:
                    # Assicura ordine logico (Economico → Standard → Premium → Luxury)
                    ORDER = ["🥉 Economico (< 7.5)", "🥈 Standard (7.5-8.5)", "🥇 Premium (8.5-9.2)", "💎 Luxury (> 9.2)"]
                    df_gap["expectation_bucket"] = df_gap["expectation_bucket"].astype(str)
                    df_gap["bucket_order"] = df_gap["expectation_bucket"].apply(lambda x: ORDER.index(x) if x in ORDER else 999)
                    df_gap = df_gap.sort_values("bucket_order").drop(columns=["bucket_order"])

                    if min_reviews > 0:
                        df_gap = df_gap[df_gap["num_reviews"] >= min_reviews]

                    if len(df_gap) == 0:
                        st.warning("Dopo il filtro, non restano fasce con abbastanza recensioni.")
                    else:
                        st.markdown("### 📊 Risultati per fascia di prestigio")

                        cols = st.columns(min(4, len(df_gap)))
                        for i, row in enumerate(df_gap.itertuples(index=False)):
                            gap_val = getattr(row, "avg_gap")
                            pct_del = getattr(row, "pct_delusioni")
                            bucket = getattr(row, "expectation_bucket")

                            # Colore delta: se gap medio è negativo -> inverse
                            delta_color = "inverse" if gap_val < 0 else "normal"

                            with cols[i % len(cols)]:
                                st.metric(
                                    label=bucket,
                                    value=f"{gap_val:.2f}",
                                    delta="Gap medio",
                                    delta_color=delta_color
                                )
                                st.caption(f"📉 Delusioni: **{pct_del:.1f}%**")
                                st.progress(min(1.0, max(0.0, pct_del / 100.0)))

                        st.divider()

                        # Grafico serio con Altair (controllo completo)
                        st.markdown("### 📉 Intensità della delusione (solo gap < 0)")
                        st.caption("Media del gap condizionata al fatto che la recensione sia negativa: più è basso, più la delusione è profonda.")

                        import altair as alt

                        chart = (
                            alt.Chart(df_gap)
                            .mark_bar()
                            .encode(
                                x=alt.X("expectation_bucket:N", sort=ORDER, title="Fascia"),
                                y=alt.Y("intensita_delusione_media:Q", title="Intensità media (gap < 0)"),
                                tooltip=[
                                    alt.Tooltip("expectation_bucket:N", title="Fascia"),
                                    alt.Tooltip("intensita_delusione_media:Q", title="Intensità", format=".3f"),
                                    alt.Tooltip("pct_delusioni:Q", title="% delusioni", format=".2f"),
                                    alt.Tooltip("num_reviews:Q", title="# recensioni")
                                ],
                            )
                            .properties(height=380)
                        )
                        st.altair_chart(chart, use_container_width=True)

                # Insight robusto (senza assumere posizioni)
                st.markdown("### 🧠 Insight")

                # Trova fascia più severa (intensità più negativa) e più “morbida”
                most_severe = df_gap.loc[df_gap["intensita_delusione_media"].idxmin()]
                least_severe = df_gap.loc[df_gap["intensita_delusione_media"].idxmax()]

                sev_bucket = most_severe["expectation_bucket"]
                sev_val = abs(most_severe["intensita_delusione_media"])

                mild_bucket = least_severe["expectation_bucket"]
                mild_val = abs(least_severe["intensita_delusione_media"])

                diff = sev_val - mild_val

                if diff >= 0.30:
                    st.warning(
                        f"⚠️ La delusione è **più intensa** nella fascia **{sev_bucket}** "
                        f"(≈ **{sev_val:.2f}** punti sotto le aspettative) rispetto alla fascia **{mild_bucket}** "
                        f"(≈ **{mild_val:.2f}**)."
                    )
                else:
                    st.info("ℹ️ L'intensità della delusione è complessivamente simile tra le fasce.")

                # Tabella (facoltativa)
                if show_table:
                    st.markdown("### 📋 Dati aggregati")
                    st.dataframe(df_gap, use_container_width=True)

    else:
        st.info("💡 Carica i dati dalla sidebar per iniziare l'analisi.")
