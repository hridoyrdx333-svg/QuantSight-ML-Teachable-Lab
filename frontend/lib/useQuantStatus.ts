"use client"

import { useEffect, useState } from "react"

export function useQuantStatus() {
  const [status, setStatus] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true

    async function load() {
      try {
        const res = await fetch("/api/status", {
          cache: "no-store",
        })

        const payload = await res.json()

        if (!alive) return

        if (!res.ok || !payload?.ok) {
          setError(payload?.error || "Backend unavailable")
          setStatus(null)
          return
        }

        setStatus(payload.data)
        setError(null)
      } catch (err) {
        if (!alive) return

        setStatus(null)
        setError(
          err instanceof Error
            ? err.message
            : "Backend unavailable"
        )
      } finally {
        if (alive) setLoading(false)
      }
    }

    load()

    const id = setInterval(load, 15000)

    return () => {
      alive = false
      clearInterval(id)
    }
  }, [])

  return {
    status,
    loading,
    error,
  }
}
