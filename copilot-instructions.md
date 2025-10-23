Scopo
In questo repository generiamo una demo locale per analizzare pattern di pagamento clienti (SAP), addestrare modelli pointwise che assegnano uno score 0–5 a ciascuna fattura aperta, e visualizzare il pattern per cliente.

Regole per Copilot/agent

Rispondi solo al task del prompt corrente. Non anticipare altri passi.

Scrivi file esatti nelle posizioni richieste, senza rinominare cartelle.

Python 3.11, src/ come package; usa type hints, docstring, logging.

I/O: leggi da data/curated/*.parquet, scrivi in data/curated, models/, reports/.

Pandas di default; niente seaborn (usa matplotlib/plotly).

Non cambiare file già creati se non esplicitamente richiesto.

Per i grafici: etichette leggibili, assi, titolo, salvataggio su disco.

Per ML: split time-based, nessun leakage; salva artefatti modello.

Accettazione: stampa report minimo indicato nel prompt; fallisci se mancano file attesi.

Dati & convenzioni

Colonne: Cliente, Data di pareggio, Doc. pareggio, Esercizio, Numero documento, Imp. div. int., Data registrazione, Data documento, Data di acquisizione, Divisa, Tipo di documento, Periodo contabile, Chiave contabile.

Chiave evento: (Cliente, Data di pareggio, Doc. pareggio).

“Aperta a t”: data_documento <= t e (data_pareggio >= t or null).

Feature: come da src/features/make_features.py (tempo/cadenza, lags, categoriali, importo relativo, profilo cliente).

Modelli: CatBoost + AutoGluon (binary prob), calibrazione Isotonic; output p_i → score 0..5.

Stile

PEP8, variabili chiare, nessuna logica “magica”.

Preferisci funzioni pure e testabili; evita side-effects.

Sicurezza & performance

Manipola i path con pathlib, nessun hard-code assoluto.

Gestisci dataset medio-grandi (100k–1M righe) con chunking se serve.

Non fare

Non implementare il combinatore somme (fuori scope).

Non toccare prompt futuri; non introdurre framework web complessi.