# 📘 Manuale Analisi Applicazione: Big Data Hotel

Questo documento descrive tutte le funzionalità di analisi implementate nell'applicazione, spiegando per ciascuna l'obiettivo, la logica tecnica ("Sotto il cofano") e i risultati mostrati.

---

## 1. 📊 Esplorazione Dati (EDA)
**Obiettivo:** Fornire una panoramica immediata sullo stato del dataset.

**Cosa Fa (Logica):**
- Carica il dataset CSV massivo in memoria distribuita usando **Apache Spark**.
- Calcola metriche aggregate su tutto il dataset in tempo reale.

**Cosa Mostra:**
- **Metriche Chiave:** Totale recensioni, Voto medio globale, Numero hotel unici.
- **Anteprima Dati:** Tabella interattiva con le prime 1000 righe per esplorare manualmente i contenuti (testo recensioni, voti, nazionalità).

---

## 2. 😊 Sentiment Analysis (Classificazione)
**Obiettivo:** Capire automaticamente se una recensione è positiva o negativa analizzando solo il testo.

**Cosa Fa (Logica):**
- **Addestramento Modello:** Utilizza un algoritmo di **Logistic Regression**.
- **Etichettatura Automatica:** Considera "Positivo" se `Reviewer_Score >= 7.5`, altrimenti "Negativo".
- **NLP Pipeline:**
  1. **Tokenizzazione:** Spezza le frasi in parole.
  2. **StopWords Removal:** Rimuove parole inutili ("the", "is", "at").
  3. **TF-IDF:** Calcola l'importanza di ogni parola (una parola rara vale di più).
  4. **Training:** Il modello impara quali parole sono associate a voti alti o bassi.

**Cosa Mostra:**
- **Accuratezza:** % di volte che il modello indovina correttamente.
- **Bilanciamento:** Grafici che mostrano quante recensioni positive vs negative ci sono.
- **Esempi Live:** Una tabella dove puoi vedere come il modello ha classificato alcune recensioni reali e con che sicurezza.

---

## 3. 🗺️ Mappa & Clustering Intelligente
Questa sezione include due visualizzazioni distinte.

### A. 📍 Mappa Geografica
**Obiettivo:** Visualizzare dove sono gli hotel e la loro qualità.
**Logica:** Aggrega latitudine/longitudine e media voto per hotel.
**Output:** Mappa interattiva dove ogni punto è un hotel.

### B. 🎯 Clustering Intelligente (K-Means)
**Obiettivo:** Raggruppare gli hotel non per posizione, ma per **caratteristiche simili** (es. "Hotel costosi ma problematici" vs "Piccoli hotel eccellenti").

**Cosa Fa (Logica):**
- **Feature Engineering:** Calcola per ogni hotel:
  - *Performance:* Voto medio, % voti eccellenti.
  - *Sentiment:* Rapporto lunghezza recensioni positive/negative.
  - *Problemi:* % recensioni che citano "construction", "dirty", "staff".
- **Algoritmo K-Means:** Divide gli hotel in K gruppi (cluster) basandosi su queste caratteristiche matematiche.
- **Auto-Naming:** Il sistema assegna automaticamente un nome al gruppo (es. "🏆 Premium Hotels") analizzando le sue statistiche medie.

**Cosa Mostra:**
- **Cards dei Gruppi:** Per ogni cluster mostra il voto medio, il numero di hotel e una descrizione interpretativa.
- **Grafici:** Confronto visivo tra i gruppi.

---

## 4. 📝 Topic Modeling (LDA)
**Obiettivo:** Scoprire **di cosa si lamentano** le persone, senza leggere migliaia di recensioni.

**Cosa Fa (Logica):**
- **Filtro:** Analizza **solo** recensioni negative lunghe (>30 caratteri).
- **Pulizia Avanzata (Regex):**
  - Mantiene solo parole di **almeno 3 lettere**.
  - Rimuove numeri, punteggiatura e simboli spazzatura (`****`, `---`).
- **Algoritmo LDA (Latent Dirichlet Allocation):** Cerca gruppi di parole che "viaggiano spesso insieme".
  - Es: Se trova spesso "room", "small", "tiny" insieme -> Crea il topic "Dimensione Stanza".

**Cosa Mostra:**
- **Lista Topic:** I temi principali emersi (es. Topic 1, Topic 2).
- **Parole Chiave:** Per ogni topic, le 7 parole più importanti con una barra di peso.

---

## 5. 🧠 Insight Avanzati (Query Specifiche)
Analisi mirate per rispondere a domande di business precise.

### 🌍 A. Nazionalità (The Grumpy Tourist)
- **Domanda:** *Quali nazioni sono più critiche o più generose?*
- **Logica:** Raggruppa per `Reviewer_Nationality` (minimo 100 recensioni).
- **Output:** Classifica "Top 5 Generosi" e "Top 5 Critici" con voto medio.

### 🏗️ B. Lavori in Corso (Construction Impact)
- **Domanda:** *Quanto paghiamo in termini di voto se ci sono lavori in corso?*
- **Logica:** Cerca parole come "construction", "renovation", "drill" nelle recensioni. Calcola il delta di voto tra chi cita lavori e chi no.
- **Output:** "I lavori riducono il voto di X punti".

### 👥 C. Tipo Viaggio (Target Analysis)
- **Domanda:** *L'hotel è meglio per coppie o famiglie?*
- **Logica:** Estrae i tag "Couple", "Family", "Solo" dalla stringa dei tag e calcola la soddisfazione media per gruppo.
- **Output:** Trofeo al target più soddisfatto.

### 📏 D. Asimmetria Emotiva (Review Length)
- **Domanda:** *La delusione genera più testo della felicità?*
- **Logica:**
  - Sfrutta i conteggi parole (`Review_Total_Negative_Word_Counts` vs `Positive`) già nel dataset.
  - Calcola il **Delta** (Differenza assoluta parole neg-pos) e il **Negativity Ratio** (solo se testo positivo rilevante).
  - Misura la **% di presenza** testo: quanto spesso il campo "Negative" o "Positive" viene riempito.
- **Output:**
  - **Metriche**: Delta medio parole per fascia di voto (es. "Arrabbiato scrive +20 parole").
  - **Grafico (Altair)**: Bar chart comparativo pulito e interattivo.
  - **Insight**: Conferma statistica dell'effetto "Sfogo" se il delta supera una soglia significativa.

### 📉 E. Affidabilità Voto (Data Consistency)
- **Domanda:** *Il voto medio è affidabile o c'è troppo disaccordo?*
- **Logica:**
  - Calcola la **Deviazione Standard (σ)** reale sui `Reviewer_Score` (misura di quanto i voti sono sparpagliati).
  - Calcola il **CV (Coefficient of Variation)** per normalizzare la dispersione rispetto alla media.
- **Output:**
  - Mappa (Altair): Scatter plot "Qualità vs Incertezza".
  - **Interpretazione**:
    - *In basso a destra*: Hotel Top e Solidi (Tutti concordano che è bello).
    - *In alto*: Hotel Rischiosi (C'è chi lo ama e chi lo odia).
    - Tooltip interattivi per esplorare ogni singolo hotel.

### ⚠️ F. Hotel Rischiosi (High Risk Detection)
- **Domanda:** *L'hotel sembra ottimo, ma nasconde scheletri nell'armadio?*
- **Logica:**
  - Filtra hotel con **Media > 8.0** (apparenza eccellente).
  - Calcola la **% Disastri** (voti ≤ 4.0).
  - **Risk Index (Smart Ranking)**: Usa la formula `Disaster% * log(NumReviews)` per dare più peso a chi ha grandi numeri (un 10% di disastri su 1000 recensioni è molto più grave che su 10).
  - Calcola il **Percentile 5% (P05)**: Il voto "minimo garantito" nel 95% dei casi.
- **Output:**
  - Lista prioritaria di "Trappole": non solo chi ha qualche voto basso, ma chi ha un *pattern* sistematico di disastri nascosto dalla media alta.
  - Grafico Scatter a 3 dimensioni (Media, % Disastri, Rischio) per isolare visivamente gli outlier pericolosi.

### 🤯 G. Expectation Gap (Realtà vs Aspettativa)
- **Domanda:** *Quanto fa male cadere dall'alto? (Aspettative deluse)*
- **Logica:**
  - Calcola il **Gap** = `Reviewer_Score` - `Average_Score` per ogni recensione.
  - Suddivide gli hotel in fasce di prestigio (Economico, Standard, Premium, Luxury).
  - Misura l'**Intensità della Delusione** (media dei Gap negativi).
- **Output:**
  - Visualizza se gli hotel di lusso vengono puniti più severamente quando sbagliano (**Paradosso del Lusso**).
  - Grafico che mostra quanto è "profondo" il disappunto per ogni fascia.
