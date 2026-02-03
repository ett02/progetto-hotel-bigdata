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
st.title("🏨 Analisi Big Data: Recensioni Hotel")
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
    ["📊 Esplorazione Dati", "😊 Sentiment Analysis", "🗺️ Mappa & Clustering", 
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
            st.dataframe(st.session_state.df_hotel.limit(1000).toPandas(), use_container_width=True)
        else:
            # Mostra solo colonne rilevanti
            columns_to_show = ["Hotel_Name", "Reviewer_Score", "Reviewer_Nationality", "Positive_Review", "Negative_Review"]
            st.dataframe(
                st.session_state.df_hotel.select(columns_to_show).limit(1000).toPandas(),
                use_container_width=True,
                height=400
            )
        
        st.markdown("")  # Spacing
    else:
        st.info("💡 Carica i dati dalla sidebar per iniziare l'esplorazione.")

# PAGINA 2: SENTIMENT ANALYSIS
elif page == "😊 Sentiment Analysis":
    st.header("😊 Sentiment Analysis (Logistic Regression)")
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
                    
                    if accuracy >= 0.75:
                        st.balloons()
                    
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
    with col2:
        st.metric("🎯 Consigliato", "3-4 topic")
    
    if st.session_state.df_hotel:
        if st.button("🚀 Estrai Topic dalle Recensioni Negative", type="primary"):
            with st.spinner("🔍 Analisi LDA in corso (può richiedere 1-2 minuti)..."):
                try:
                    result = gestore.esegui_topic_modeling(st.session_state.df_hotel, num_topics=num_topics)
                    
                    # Unwrap results
                    topics_data = result['topics_data']
                    vocab = result['vocab']
                    num_reviews = result['num_reviews']
                    log_perplexity = result['log_perplexity']
                    
                    st.success(f"✅ Analisi completata su **{num_reviews:,}** recensioni negative!")
                    
                    # Metriche di qualità
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("📊 Recensioni Analizzate", f"{num_reviews:,}")
                    with col2:
                        st.metric("📉 Log-Perplexity", f"{log_perplexity:.2f}", 
                                 help="Misura la qualità del modello. Valori più bassi indicano modelli migliori.")
                    
                    st.divider()
                    st.subheader("🗂️ Topic Trovati")
                    st.caption("Ogni topic è un gruppo di parole correlate che rappresentano un tema comune nelle lamentele")
                    
                    # Visualizzazione Topic con cards
                    topics_list = topics_data.collect()
                    
                    # Usa colonne per layout più pulito
                    for row in topics_list:
                        topic_id = row['topic']
                        term_indices = row['termIndices']
                        term_weights = row['termWeights']
                        
                        # Estrai parole e pesi
                        terms_with_weights = [(vocab[idx], float(weight)) for idx, weight in zip(term_indices, term_weights)]
                        
                        # Card per ogni topic
                        with st.container():
                            st.markdown(f"### 📌 Topic {topic_id + 1}")
                            
                            # Mostra top 7 termini con barre di peso
                            top_terms = terms_with_weights[:7]
                            
                            for term, weight in top_terms:
                                # Crea barra di progresso visuale per il peso
                                normalized_weight = min(weight * 100, 100)  # Normalizza per visualizzazione
                                st.progress(normalized_weight / 100, text=f"**{term}** ({weight:.3f})")
                            
                            st.caption(f"💡 **Interpretazione suggerita**: {', '.join([t[0] for t in top_terms[:3]])}")
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
            ["🌍 Nazionalità", "🏗️ Lavori in Corso", "👥 Tipo Viaggio"],
            help="Seleziona il tipo di analisi avanzata da eseguire"
        )
        
        st.markdown("")  # Spacing
        st.divider()
        
        # ========= QUERY 1: NAZIONALITÀ =========
        if "Nazionalità" in query_type:
            st.subheader("🌍 Analisi per Nazionalità")
            st.markdown("""
            **Obiettivo**: Scoprire quali nazionalità danno mediamente i **voti più alti** o **più bassi**.  
            **Utilità**: Capire le aspettative culturali e targetizzare meglio il marketing.
            """)
            
            with st.expander("ℹ️ Come funziona"):
                st.markdown("""
                - **Filtraggio**: Solo nazionalità con >100 recensioni (per affidabilità statistica)
                - **Metriche**: Voto medio, deviazione standard, min/max
                - **Interpretazione**: 
                  - **Voto alto** → Turisti soddisfatti, aspettative moderate
                  - **Voto basso** → Turisti esigenti, standard elevati
                  - **Alta deviazione** → Opinioni molto diverse
                """)
            
            if st.button("🚀 Esegui Analisi Nazionalità", type="primary"):
                with st.spinner("Analisi in corso..."):
                    df_naz = gestore.query_nazionalita_critiche(st.session_state.df_hotel).toPandas()
                    
                    if len(df_naz) > 0:
                        # Metriche generali
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📌 Nazionalità Analizzate", len(df_naz))
                        with col2:
                            std_media = df_naz['deviazione_std'].mean()
                            st.metric("📊 Deviazione Std Media", f"{std_media:.2f}")
                        with col3:
                            range_voti = df_naz['voto_medio'].max() - df_naz['voto_medio'].min()
                            st.metric("📈 Range Voti", f"{range_voti:.2f}")
                        
                        st.divider()
                        
                        # Top/Bottom 5
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 😊 Top 5 - Più Generosi")
                            top5 = df_naz.nlargest(5, 'voto_medio')[['Reviewer_Nationality', 'voto_medio', 'num_recensioni']]
                            top5.columns = ['Nazionalità', 'Voto Medio', 'Recensioni']
                            st.dataframe(top5, use_container_width=True, hide_index=True)
                        
                        with col2:
                            st.markdown("### 😤 Top 5 - Più Critici")
                            bottom5 = df_naz.nsmallest(5, 'voto_medio')[['Reviewer_Nationality', 'voto_medio', 'num_recensioni']]
                            bottom5.columns = ['Nazionalità', 'Voto Medio', 'Recensioni']
                            st.dataframe(bottom5, use_container_width=True, hide_index=True)
                        
                        # Grafico a barre (top 15)
                        st.markdown("### 📊 Confronto Visivo (Top 15 + Bottom 15)")
                        import pandas as pd
                        df_viz = pd.concat([df_naz.head(15), df_naz.tail(15)])
                        st.bar_chart(df_viz.set_index('Reviewer_Nationality')['voto_medio'])
                    else:
                        st.warning("Nessun dato trovato per questa analisi.")
        
        # ========= QUERY 2: COSTRUZIONI =========
        elif "Lavori" in query_type:
            st.subheader("🏗️ Impatto dei Lavori in Corso")
            st.markdown("""
            **Obiettivo**: Quantificare quanto i **lavori di ristrutturazione/costruzione** impattano negativamente sul voto.  
            **Utilità**: Informare i clienti in anticipo e gestire le aspettative.
            """)
            
            with st.expander("ℹ️ Come funziona"):
                st.markdown("""
                - **Keywords**: construction, renovation, works, hammering, drilling, noise, building
                - **Confronto**: Voto medio con lavori VS senza lavori
                - **Interpretazione**:
                  - **Differenza negativa** → I lavori riducono la soddisfazione
                  - **Alta deviazione** → Esperienza molto variabile
                """)
            
            if st.button("🚀 Esegui Analisi Lavori", type="primary"):
                with st.spinner("Cercando recensioni con menzioni di lavori..."):
                    df_cost = gestore.query_impatto_costruzioni(st.session_state.df_hotel).toPandas()
                    
                    if len(df_cost) >= 2:
                        # Estrai dati
                        no_lavori = df_cost[df_cost['has_construction'] == False]
                        si_lavori = df_cost[df_cost['has_construction'] == True]
                        
                        voto_no = no_lavori['voto_medio'].values[0] if len(no_lavori) > 0 else 0
                        voto_si = si_lavori['voto_medio'].values[0] if len(si_lavori) > 0 else 0
                        count_si = si_lavori['totale'].values[0] if len(si_lavori) > 0 else 0
                        
                        diff = voto_si - voto_no
                        
                        # Metriche
                        col1, col2,col3 = st.columns(3)
                        with col1:
                            st.metric("🏨 Senza Lavori", f"{voto_no:.2f}", help="Voto medio quando non ci sono lavori")
                        with col2:
                            st.metric("🏗️ Con Lavori", f"{voto_si:.2f}", 
                                     delta=f"{diff:.2f}" if diff < 0 else f"+{diff:.2f}",
                                     delta_color="inverse")
                        with col3:
                            st.metric("📊 Recensioni con Lavori", f"{count_si:,}")
                        
                        # Interpretazione
                        if diff < -0.3:
                            st.error(f"⚠️ **IMPATTO SIGNIFICATIVO**: I lavori riducono il voto di **{abs(diff):.2f} punti**. Consigliato informare i clienti in anticipo.")
                        elif diff < 0:
                            st.warning(f"🔸 **Impatto moderato**: I lavori riducono il voto di **{abs(diff):.2f} punti**.")
                        else:
                            st.success("✅ Nessun impatto negativo rilevato.")
                        
                        # Grafico
                        st.bar_chart(df_cost.set_index('has_construction')['voto_medio'])
                    else:
                        st.warning("Dati insufficienti per il confronto.")
        
        # ========= QUERY 3: TIPO VIAGGIO =========
        else:
            st.subheader("👥 Analisi per Tipo di Viaggio")
            st.markdown("""
            **Obiettivo**: Capire quale **target** (coppie, famiglie, solitari) è più soddisfatto.  
            **Utilità**: Ottimizzare servizi e marketing per il target giusto.
            """)
            
            with st.expander("ℹ️ Come funziona"):
                st.markdown("""
                - **Categorie**: Coppia, Famiglia, Solo, Gruppo, Altro
                - **Filtro**: Solo gruppi con >50 recensioni
                - **Interpretazione**:
                  - **Voto alto** → Target soddisfatto
                  - **Bassa deviazione** → Esperienza consistente
                """)
            
            if st.button("🚀 Esegui Analisi Tipo Viaggio", type="primary"):
                with st.spinner("Estraendo tag di viaggio..."):
                    df_viaggi = gestore.query_coppie_vs_famiglie(st.session_state.df_hotel).toPandas()
                    
                    if len(df_viaggi) > 0:
                        # Metriche
                        st.markdown("### 📊 Risultati per Categoria")
                        for idx, row in df_viaggi.iterrows():
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.metric(row['tipo_viaggio'], f"{row['voto_medio']:.2f} ⭐")
                            with col2:
                                st.caption(f"📝 {row['num_recensioni']:,} recensioni")
                            with col3:
                                st.caption(f"📊 σ={row['deviazione_std']:.2f}")
                        
                        st.divider()
                        
                        # Grafico
                        st.markdown("### 📈 Confronto Visivo")
                        st.bar_chart(df_viaggi.set_index('tipo_viaggio')['voto_medio'])
                        
                        # Insight
                        best = df_viaggi.iloc[0]
                        st.success(f"🏆 **Target più soddisfatto**: {best['tipo_viaggio']} con {best['voto_medio']:.2f}/10")
                    else:
                        st.warning("Nessun dato trovato.")
    else:
        st.info("💡 Carica i dati dalla sidebar per iniziare l'analisi.")
