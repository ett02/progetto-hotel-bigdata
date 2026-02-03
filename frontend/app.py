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
            ["🌍 Nazionalità", "🏗️ Lavori in Corso", "👥 Tipo Viaggio", "📏 Lunghezza Recensioni", "📉 Affidabilità Voto (Std Dev)", "⚠️ Hotel Rischiosi (Alto Rischio)", "🤯 Expectation Gap (Realtà vs Aspettativa)"],
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
        elif "Tipo Viaggio" in query_type:
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
                    # --- Ordine logico bucket (non fidarti dell’ordine arrivato) ---
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
                            # metrica principale: delta (più interpretabile)
                            st.metric(
                                label=bucket,
                                value=f"{delta:.1f}",
                                delta="Δ (neg - pos) parole",
                                delta_color="inverse" if delta > 0 else "normal",
                                help=f"Neg: {neg_len:.0f} parole | Pos: {pos_len:.0f} parole | Ratio: {ratio_str}"
                            )

                            # secondaria: ratio (se disponibile)
                            st.caption(f"Ratio (neg/pos): **{ratio_str}**")

                            # presenza testo (se disponibile)
                            pn = getattr(row, "pct_has_negative", None)
                            pp = getattr(row, "pct_has_positive", None)
                            if pn is not None and pp is not None:
                                st.caption(f"Testo Neg presente: **{float(pn):.1f}%** | Pos presente: **{float(pp):.1f}%**")

                    st.divider()

                    # --- Grafico serio con Altair (affidabile) ---
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

                    # --- Insight robusto (senza iloc[0]/[-1]) ---
                    st.markdown("### 🧠 Insight automatico")

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
                st.markdown("### 🧠 Insight automatico")

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
