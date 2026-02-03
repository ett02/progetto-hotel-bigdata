import streamlit as st
import pandas as pd
import sys
import os

# Aggiunge la cartella 'backend' al path per permettere l'import del modulo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from backend import GestoreBigData

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

# Tab Layout
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Esplorazione Dati (EDA)", "😊 Sentiment Analysis", "🌍 Clustering Hotel", "📝 Topic Modeling", "🧠 Insight Avanzati"])

# TAB 1: EDA
with tab1:
    st.header("Esplorazione Dati")
    if st.session_state.df_hotel:
        st.subheader("Anteprima Dataset Hotel")
        # Convertiamo in Pandas solo una piccola parte per la visualizzazione
        st.dataframe(st.session_state.df_hotel.limit(1000).toPandas())
        
        st.subheader("Statistiche Veloci")
        num_hotel = st.session_state.df_hotel.count()
        avg_score = st.session_state.df_hotel.select("Average_Score").agg({"Average_Score": "avg"}).collect()[0][0]
        st.metric("Totale Recensioni", num_hotel)
        st.metric("Punteggio Medio Globale", f"{avg_score:.2f}")
    else:
        st.info("Carica i dati dalla sidebar per iniziare.")

# TAB 2: SENTIMENT ANALYSIS
with tab2:
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

# TAB 3: CLUSTERING
with tab3:
    st.header("Clustering Geografico e Qualitativo (K-Means)")
    st.markdown("Raggruppa gli hotel in base a posizione e punteggio.")
    
    k_value = st.slider("Numero di Cluster (K)", 2, 10, 5)
    
    if st.session_state.df_hotel:
        if st.button("Esegui Clustering"):
            with st.spinner("Esecuzione K-Means..."):
                risultati_km, silhouette = gestore.esegui_clustering_hotel(st.session_state.df_hotel, k=k_value)
                st.metric("Silhouette Score (Qualità Cluster)", f"{silhouette:.4f}")
                
                # Visualizzazione Mappa
                # Convertiamo tutto il DF in Pandas per st.map (attenzione: se è troppo grande potrebbe rallentare, 
                # ma per gli hotel unici dovrebbe andare bene. Qui usiamo tutte le recensioni, che è pesante.
                # Meglio raggruppare per hotel prima di visualizzare)
                df_map = risultati_km.select("lat", "lng", "prediction").sample(fraction=0.1).toPandas() # Campionamento per velocità
                
                # FIX: Streamlit richiede 'longitude' invece di 'lng'
                df_map = df_map.rename(columns={'lng': 'longitude'})
                
                # Rinominiamo prediction in color o usiamo st.map semplice
                st.map(df_map)
                st.caption("Mappa di un campione del 10% delle recensioni, colorate per cluster.")
    else:
        st.info("Carica i dati prima di procedere.")

# TAB 4: TOPIC MODELING
with tab4:
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
        num_topics = st.slider("🔢 Numero di Topic da estrarre", 2, 6, 3, help="Più topic = più granularità, ma rischio di sovrapposizione")
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


# TAB 5: INSIGHT AVANZATI
with tab5:
    st.header("🧠 Insight Avanzati - Query Spark Personalizzate")
    st.markdown("""
    Analisi avanzate basate su **query Spark ottimizzate** per rivelare pattern nascosti nei dati.
    Esplora comportamenti per nazionalità, impatto dei lavori, e preferenze per tipo di viaggio.
    """)
    
    if st.session_state.df_hotel:
        # Selezione query
        query_type = st.selectbox(
            "📊 Seleziona un'analisi",
            ["🌍 Nazionalità: Chi sono i turisti più critici?", 
             "🏗️ Lavori in Corso: Quanto impattano sul voto?",
             "👥 Tipo Viaggio: Coppie vs Famiglie vs Solo"]
        )
        
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
