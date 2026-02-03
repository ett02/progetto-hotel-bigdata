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
  - Suddivide le recensioni in 4 bucket: *Arrabbiato* (<5), *Deluso* (5-7.5), *Soddisfatto* (7.5-9), *Felice* (>9).
  - Somma il numero di parole negative (`Negative_Review`) e positive (`Positive_Review`) per ogni gruppo.
  - Calcola il **Negativity Ratio**: Quanto è più lunga la parte negativa rispetto alla positiva?
- **Output:**
  - **Metriche**: Ratio per ogni fascia (es. "3.5x" significa che scrivono 3 volte e mezzo in più).
  - **Grafico**: Barre che confrontano visivamente la "voglia di scrivere" in negativo (Rosso) vs positivo (Verde).
  - **Insight**: Conferma statistica se l'hotel soffre dell'effetto "Sfogo" (Clienti che scrivono papiri solo per lamentarsi).

### 📉 E. Affidabilità Voto (Data Consistency)
- **Domanda:** *Il voto medio è affidabile o c'è troppo disaccordo?*
- **Logica:**
  - Calcola la **Deviazione Standard** dei voti per ogni hotel.
  - Bassa deviazione = Opinioni coerenti (Affidabile).
  - Alta deviazione = Opinioni polarizzate "Love or Hate" (Rischioso).
- **Output:** Mappa del rischio (Scatter plot) che mostra la relazione tra Qualità e Incertezza.

### ⚠️ F. Hotel Rischiosi (High Risk Detection)
- **Domanda:** *L'hotel sembra ottimo, ma nasconde scheletri nell'armadio?*
- **Logica:**
  - Filtra hotel con **Media > 8.0** (apparentemente eccellenti).
  - Ma con **> 5% di recensioni disastrose** (voto <= 4.0).
- **Output:** Lista di "Trappole potenziali": hotel con media alta ma probabilità elevata di pessima esperienza.
