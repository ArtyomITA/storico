""" SAP FI-AR Glossary - Terminologia corretta italiana per UI."""
from __future__ import annotations

from typing import Dict, Any

# ================================================================================================
# SAP FI-AR (Financial Accounting - Accounts Receivable) Glossary
# Fonte: SAP Help Portal, SAP Fiori Apps Reference Library, SAP Business One
# ================================================================================================

SAP_FI_AR_GLOSSARY: Dict[str, Dict[str, Any]] = {
    # --- Documenti Contabili ---
    "partite_aperte": {
        "term_it": "Partite Aperte",
        "term_sap_en": "Open Items",
        "term_sap_code": "BSID / BSAD",
        "definition": "Documenti contabili (fatture, note di credito) che non sono ancora stati completamente compensati o pagati.",
        "context": "Una fattura cliente rimane 'aperta' fino a quando non viene registrato il pagamento corrispondente.",
        "ui_label": "Partite Aperte",
        "tooltip": "Documenti non ancora compensati (SAP: Open Items)",
    },
    "documento_compensazione": {
        "term_it": "Documento di Compensazione",
        "term_sap_en": "Clearing Document",
        "term_sap_code": "AUGBL",
        "definition": "Il documento contabile che registra il pagamento ricevuto e compensa (chiude) le partite aperte.",
        "context": "Quando un cliente paga una fattura, SAP genera un documento di compensazione che 'pareggia' la partita.",
        "ui_label": "Doc. Compensazione",
        "tooltip": "Documento che registra il pagamento e chiude la partita aperta (SAP: Clearing Document)",
    },
    "compensazione": {
        "term_it": "Compensazione",
        "term_sap_en": "Clearing",
        "term_sap_code": "FB05 / F-28",
        "definition": "Il processo di chiusura di una o più partite aperte mediante registrazione del pagamento ricevuto.",
        "context": "La compensazione può essere automatica (tramite bank statement) o manuale (tramite transazione FB05).",
        "ui_label": "Compensazione",
        "tooltip": "Chiusura partite aperte tramite registrazione pagamento (SAP: Clearing)",
    },
    "data_compensazione": {
        "term_it": "Data di Compensazione",
        "term_sap_en": "Clearing Date",
        "term_sap_code": "AUGDT",
        "definition": "La data in cui è stata effettuata la compensazione (chiusura) della partita aperta.",
        "context": "Può differire dalla data documento se il pagamento viene registrato in ritardo.",
        "ui_label": "Data Compensazione",
        "tooltip": "Data di chiusura della partita aperta (SAP: Clearing Date - AUGDT)",
    },
    # --- Date Contabili ---
    "data_documento": {
        "term_it": "Data Documento",
        "term_sap_en": "Document Date",
        "term_sap_code": "BLDAT",
        "definition": "La data riportata sul documento originale (es. data fattura stampata sul documento).",
        "context": "Usata per ordinamento cronologico documenti, può differire dalla data di registrazione.",
        "ui_label": "Data Documento",
        "tooltip": "Data riportata sul documento originale (SAP: Document Date - BLDAT)",
    },
    "data_registrazione": {
        "term_it": "Data di Registrazione",
        "term_sap_en": "Posting Date",
        "term_sap_code": "BUDAT",
        "definition": "La data in cui il documento viene registrato nel libro mastro SAP.",
        "context": "Determina il periodo contabile di competenza. Può essere retroattiva se periodo ancora aperto.",
        "ui_label": "Data Registrazione",
        "tooltip": "Data di registrazione nel libro mastro (SAP: Posting Date - BUDAT)",
    },
    "data_base_pagamento": {
        "term_it": "Data Base Pagamento",
        "term_sap_en": "Baseline Date for Payment",
        "term_sap_code": "ZFBDT",
        "definition": "La data di riferimento per il calcolo della scadenza pagamento. Di default = data documento.",
        "context": "Usata in combinazione con i termini di pagamento (es. 30gg dalla data base) per calcolare la scadenza.",
        "ui_label": "Data Base Pagamento",
        "tooltip": "Data di riferimento per calcolo scadenza (SAP: Baseline Date - ZFBDT)",
    },
    "data_scadenza": {
        "term_it": "Data di Scadenza",
        "term_sap_en": "Due Date / Net Due Date",
        "term_sap_code": "ZFBDT + ZBD*T",
        "definition": "La data entro cui il pagamento deve essere effettuato secondo i termini di pagamento concordati.",
        "context": "Calcolata come: Data Base Pagamento + Giorni Termini Pagamento.",
        "ui_label": "Data Scadenza",
        "tooltip": "Data entro cui il pagamento è dovuto (SAP: Net Due Date)",
    },
    # --- Chiavi e Codici ---
    "chiave_contabile": {
        "term_it": "Chiave Contabile",
        "term_sap_en": "Posting Key",
        "term_sap_code": "BSCHL",
        "definition": "Codice che determina il tipo di operazione contabile (dare/avere) e il tipo di conto (clienti, fornitori, CoGe).",
        "context": "Es. 01=Dare CoGe, 50=Dare Clienti, 40=Avere Clienti, 31=Avere Fornitori.",
        "ui_label": "Chiave Contabile",
        "tooltip": "Codice operazione contabile dare/avere (SAP: Posting Key - BSCHL)",
    },
    "tipo_documento": {
        "term_it": "Tipo Documento",
        "term_sap_en": "Document Type",
        "term_sap_code": "BLART",
        "definition": "Codice che classifica il tipo di documento contabile (fattura, nota credito, pagamento, storno).",
        "context": "Es. DR=Fattura Cliente, DG=Nota Credito Cliente, DZ=Pagamento Cliente, ZP=Pagamento Fornitore.",
        "ui_label": "Tipo Documento",
        "tooltip": "Classificazione documento contabile (SAP: Document Type - BLART)",
    },
    "numero_documento": {
        "term_it": "Numero Documento",
        "term_sap_en": "Document Number",
        "term_sap_code": "BELNR",
        "definition": "Identificativo univoco del documento contabile nel sistema SAP.",
        "context": "Combinato con codice società (BUKRS) e anno fiscale (GJAHR) forma la chiave primaria.",
        "ui_label": "Num. Documento",
        "tooltip": "Identificativo univoco documento (SAP: Document Number - BELNR)",
    },
    # --- Importi e Valute ---
    "importo_documento": {
        "term_it": "Importo Documento",
        "term_sap_en": "Document Amount",
        "term_sap_code": "DMBTR",
        "definition": "L'importo totale del documento nella valuta della società.",
        "context": "Se fattura in valuta estera, convertito in valuta locale al tasso di cambio della data documento.",
        "ui_label": "Importo",
        "tooltip": "Importo totale documento (SAP: Document Amount - DMBTR)",
    },
    "importo_in_valuta_locale": {
        "term_it": "Importo in Valuta Locale",
        "term_sap_en": "Amount in Local Currency",
        "term_sap_code": "DMBTR",
        "definition": "Importo convertito nella valuta della società (es. EUR per società italiana).",
        "context": "Usato per reporting consolidato e analisi indipendenti dalla valuta transazione.",
        "ui_label": "Importo (Valuta Locale)",
        "tooltip": "Importo convertito in valuta locale (SAP: Local Currency Amount)",
    },
    "importo_in_valuta_documento": {
        "term_it": "Importo in Valuta Documento",
        "term_sap_en": "Amount in Document Currency",
        "term_sap_code": "WRBTR",
        "definition": "Importo originale nella valuta del documento (es. USD per fattura export).",
        "context": "Preserva l'importo originale della transazione, utile per riconciliazioni con estratti conto esteri.",
        "ui_label": "Importo (Valuta Doc.)",
        "tooltip": "Importo originale nella valuta del documento (SAP: Document Currency Amount - WRBTR)",
    },
    # --- Clienti e Conti ---
    "conto_cliente": {
        "term_it": "Conto Cliente",
        "term_sap_en": "Customer Account",
        "term_sap_code": "KUNNR",
        "definition": "Codice univoco che identifica il cliente nel sistema SAP.",
        "context": "Gestito in tabella KNA1 (dati generali) e KNB1 (dati società-specifici).",
        "ui_label": "Cliente",
        "tooltip": "Codice identificativo cliente (SAP: Customer Account - KUNNR)",
    },
    "partita_cliente": {
        "term_it": "Partita Cliente",
        "term_sap_en": "Customer Line Item",
        "term_sap_code": "BSID / BSAD",
        "definition": "Singola riga contabile intestata a un conto cliente (fattura, pagamento, nota credito).",
        "context": "BSID = partite aperte, BSAD = partite compensate (archiviate).",
        "ui_label": "Partita Cliente",
        "tooltip": "Riga contabile intestata a cliente (SAP: Customer Line Item)",
    },
    # --- Pattern di Pagamento (Gergo Interno → SAP) ---
    "evento_pagamento": {
        "term_it": "Evento di Pagamento",
        "term_sap_en": "Payment Event / Clearing Event",
        "term_sap_code": "AUGBL (grouped by AUGDT + KUNNR)",
        "definition": "Un gruppo di partite aperte che vengono compensate insieme dallo stesso cliente nella stessa data.",
        "context": "Corrisponde a un bonifico o un'operazione di cassa che chiude multiple fatture simultaneamente.",
        "ui_label": "Evento di Pagamento",
        "tooltip": "Gruppo di partite compensate insieme (stesso cliente, stessa data)",
    },
    "cadenza_pagamento": {
        "term_it": "Cadenza di Pagamento",
        "term_sap_en": "Payment Frequency / Payment Cadence",
        "term_sap_code": "N/A (derived metric)",
        "definition": "La frequenza con cui un cliente effettua pagamenti (settimanale, quindicinale, mensile, fine mese).",
        "context": "Calcolato analizzando l'intervallo medio tra eventi di pagamento consecutivi.",
        "ui_label": "Cadenza Pagamento",
        "tooltip": "Frequenza pagamenti cliente (es. settimanale, mensile)",
    },
    "regolarita_pagamento": {
        "term_it": "Regolarità di Pagamento",
        "term_sap_en": "Payment Regularity / Payment Consistency",
        "term_sap_code": "N/A (derived metric)",
        "definition": "Misura della coerenza temporale dei pagamenti rispetto alla cadenza tipica del cliente.",
        "context": "Es. Cliente con cadenza mensile che paga sempre entro ±3 giorni dal giorno tipico = alta regolarità.",
        "ui_label": "Regolarità",
        "tooltip": "Coerenza temporale dei pagamenti rispetto alla cadenza tipica",
    },
    "ritardo_medio": {
        "term_it": "Ritardo Medio Pagamento",
        "term_sap_en": "Average Payment Delay / Days Sales Outstanding (DSO)",
        "term_sap_code": "N/A (derived: AUGDT - ZFBDT - payment_terms)",
        "definition": "Il numero medio di giorni tra la scadenza pagamento e la data effettiva di compensazione.",
        "context": "Positivo = ritardo, Negativo = anticipo. DSO = Days Sales Outstanding (metrica standard FI).",
        "ui_label": "Ritardo Medio",
        "tooltip": "Giorni medi di ritardo rispetto alla scadenza (DSO)",
    },
    # --- Policy di Pagamento (Gergo Interno → SAP Friendly) ---
    "policy_settimanale": {
        "term_it": "Pagamento Settimanale",
        "term_sap_en": "Weekly Payment Pattern",
        "term_sap_code": "N/A (pattern detector)",
        "definition": "Cliente che effettua pagamenti con cadenza di circa 7 giorni.",
        "context": "Tipico di clienti retail o piccole imprese con cash flow settimanale.",
        "ui_label": "Settimanale",
        "tooltip": "Pagamenti con cadenza ~7 giorni",
    },
    "policy_quindicinale": {
        "term_it": "Pagamento Quindicinale",
        "term_sap_en": "Bi-weekly Payment Pattern",
        "term_sap_code": "N/A (pattern detector)",
        "definition": "Cliente che effettua pagamenti con cadenza di circa 14 giorni.",
        "context": "Tipico di clienti con cicli di cassa quindicinali (es. stipendi quindicinali).",
        "ui_label": "Quindicinale",
        "tooltip": "Pagamenti con cadenza ~14 giorni",
    },
    "policy_mensile": {
        "term_it": "Pagamento Mensile",
        "term_sap_en": "Monthly Payment Pattern",
        "term_sap_code": "N/A (pattern detector)",
        "definition": "Cliente che effettua pagamenti con cadenza di circa 30 giorni.",
        "context": "Pattern più comune, allineato a cicli mensili di fatturazione e chiusura contabile.",
        "ui_label": "Mensile",
        "tooltip": "Pagamenti con cadenza ~30 giorni",
    },
    "policy_fine_mese": {
        "term_it": "Pagamento a Fine Mese",
        "term_sap_en": "End-of-Month Payment Pattern",
        "term_sap_code": "N/A (pattern detector)",
        "definition": "Cliente che concentra i pagamenti negli ultimi giorni del mese.",
        "context": "Tipico di grandi aziende che processano pagamenti in batch a fine mese per controllo cassa.",
        "ui_label": "Fine Mese",
        "tooltip": "Pagamenti concentrati negli ultimi giorni del mese",
    },
    "policy_pagamento_totale": {
        "term_it": "Pagamento Totale Fattura",
        "term_sap_en": "Full Invoice Payment",
        "term_sap_code": "N/A (pattern detector)",
        "definition": "Cliente che compensa sempre l'intero importo della fattura (no pagamenti parziali).",
        "context": "Opposto a pagamenti parziali o rateizzati. Indica buona salute finanziaria.",
        "ui_label": "Pagamento Totale",
        "tooltip": "Compensa sempre l'intero importo fattura (no pagamenti parziali)",
    },
    "policy_regmonth": {
        "term_it": "Mese Registrazione = Mese Compensazione",
        "term_sap_en": "Same Month Posting and Clearing",
        "term_sap_code": "N/A (pattern detector: MONTH(BUDAT) = MONTH(AUGDT))",
        "definition": "Cliente che compensa le partite nello stesso mese di registrazione del documento.",
        "context": "Non significa 'pagato entro il mese corrente' ma 'data registrazione e data compensazione nello stesso mese'.",
        "ui_label": "Stesso Mese (Reg=Comp)",
        "tooltip": "Compensazione nello stesso mese della registrazione documento (non 'pagato entro il mese')",
    },
    # --- Clustering e Segmentazione ---
    "cluster_clienti": {
        "term_it": "Cluster Clienti",
        "term_sap_en": "Customer Cluster / Customer Segment",
        "term_sap_code": "N/A (analytics)",
        "definition": "Gruppo di clienti con comportamenti di pagamento simili, identificato mediante algoritmi di clustering.",
        "context": "Es. Cluster 'Pagatori Puntuali', 'Ritardatari Cronici', 'Pagatori Fine Mese'.",
        "ui_label": "Cluster",
        "tooltip": "Gruppo clienti con comportamenti simili",
    },
    "profilo_cluster": {
        "term_it": "Profilo Cluster",
        "term_sap_en": "Cluster Profile / Segment Profile",
        "term_sap_code": "N/A (analytics)",
        "definition": "Descrizione statistica delle caratteristiche medie di un cluster (cadenza, ritardo, regolarità).",
        "context": "Usato per etichettare cluster e personalizzare strategie di gestione crediti.",
        "ui_label": "Profilo",
        "tooltip": "Caratteristiche medie del cluster",
    },
    # --- KPI e Metriche ---
    "kpi_affidabilita": {
        "term_it": "Affidabilità Pagamento",
        "term_sap_en": "Payment Reliability / Payment Score",
        "term_sap_code": "N/A (derived metric)",
        "definition": "Indice che misura la probabilità che un cliente paghi entro la scadenza concordata.",
        "context": "Basato su storico pagamenti: % pagamenti puntuali, ritardo medio, regolarità.",
        "ui_label": "Affidabilità",
        "tooltip": "Probabilità di pagamento puntuale (0-100%)",
    },
    "kpi_esposizione": {
        "term_it": "Esposizione Credito",
        "term_sap_en": "Credit Exposure / Receivables Outstanding",
        "term_sap_code": "BSID sum(DMBTR)",
        "definition": "Importo totale delle partite aperte per un cliente (debito residuo).",
        "context": "Somma di tutte le fatture non ancora compensate. Rilevante per valutazione rischio credito.",
        "ui_label": "Esposizione",
        "tooltip": "Importo totale partite aperte (SAP: Receivables Outstanding)",
    },
    "kpi_dso": {
        "term_it": "DSO (Days Sales Outstanding)",
        "term_sap_en": "Days Sales Outstanding",
        "term_sap_code": "N/A (derived: AVG(AUGDT - invoice_date))",
        "definition": "Numero medio di giorni necessari per incassare una fattura dopo l'emissione.",
        "context": "KPI standard FI. Target tipico: 30-60 giorni. Valori alti indicano problemi di cash collection.",
        "ui_label": "DSO",
        "tooltip": "Giorni medi per incassare fatture (Days Sales Outstanding)",
    },
}


def get_sap_term(key: str, field: str = "ui_label") -> str:
    """Get SAP terminology from glossary."""
    return SAP_FI_AR_GLOSSARY.get(key, {}).get(field, key)


def get_sap_tooltip(key: str) -> str:
    """Get SAP tooltip from glossary."""
    return SAP_FI_AR_GLOSSARY.get(key, {}).get("tooltip", "")


def render_inline_glossary_icon(term_key: str) -> str:
    """Render inline glossary icon with tooltip (future: link to full glossary page)."""
    tooltip = get_sap_tooltip(term_key)
    if not tooltip:
        return ""
    # Return markdown with info icon + tooltip
    # Future enhancement: make icon clickable to open glossary page
    return f" ℹ️"  # Placeholder for now, will be enhanced with st.help() or custom component


# ================================================================================================
# Policy Labels (Italian) - for Pattern Summary UI
# ================================================================================================
POLICY_LABELS_IT: Dict[str, str] = {
    "fifo": "FIFO (Più Vecchio)",
    "lifo": "LIFO (Più Recente)",
    "largest": "Largest (Importo Max)",
    "smallest": "Smallest (Importo Min)",
    "same_period": "Stesso Periodo",
    "eomfirst": "Fine Mese",
    "regmonth": "Stesso Mese Registrazione",
    "quarter": "Stesso Trimestre",
    "seqblock": "Blocco Sequenziale",
    "typeprio": "Priorità per Tipo",
    "unknown": "Sconosciuto",
}
