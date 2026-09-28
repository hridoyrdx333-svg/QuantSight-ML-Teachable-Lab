# QuantSight UI polished build

This build keeps the original QuantSight dashboard visual language but upgrades all secondary pages.

Updated:
- Market: richer asset cards, comparison table, visualization area
- AI Model: thresholds, runtime contract, diagnostics workspace
- Signals: KPI row, filters, ledger table
- Paper Trading: risk limits, safety controls, execution ledger
- Outcomes: evaluation metrics and ledger
- Risk: guard status and safety-event panel
- System Health: service cards, pipeline view, freshness/details, status timeline
- Data & Evidence: artifact cards, index table, audit notes
- Settings: compact two-column preference layout

Safety/semantics:
- No real API keys or credentials added
- No backend/model/trading logic changed
- No fake profit or accuracy values added
- Approved Model remains NOT READY
- Demo-only / Orders OFF messaging preserved
- PLACE_ORDERS is not enabled

Next step:
Connect the frontend to the existing QuantSight Flask APIs and replace API-pending values with live runtime data.
