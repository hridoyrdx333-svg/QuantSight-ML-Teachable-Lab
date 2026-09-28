export function probabilityText(value: any) {
  const n = Number(value)

  if (!Number.isFinite(n)) return "—"

  return `${(n * 100).toFixed(2)}%`
}

export function ageText(value: any) {
  const n = Number(value)

  if (!Number.isFinite(n)) return "—"

  if (n < 60) return `${Math.round(n)} sec`

  if (n < 3600) {
    return `${Math.round(n / 60)} min`
  }

  return `${(n / 3600).toFixed(1)} hr`
}

export function getSignalRows(status: any) {
  const latest = status?.shadow?.latest || {}

  const thresholds =
    status?.shadow?.status?.thresholds || {}

  return ["BTCUSDT", "ETHUSDT", "SOLUSDT"].map(
    (symbol) => {
      const s = latest[symbol] || {}

      return {
        symbol,

        probability:
          probabilityText(s.probability_up),

        threshold:
          s.locked_threshold ??
          thresholds[symbol] ??
          (symbol === "SOLUSDT" ? 0.54 : 0.56),

        decision:
          s.decision || "—",

        featureAge:
          ageText(s.feature_age_seconds),

        fresh:
          status?.shadow?.fresh
            ? "Fresh"
            : s.decision === "STALE"
            ? "Stale"
            : "Pending",

        orders:
          s.orders_sent ? "SENT" : "OFF",
      }
    }
  )
}
