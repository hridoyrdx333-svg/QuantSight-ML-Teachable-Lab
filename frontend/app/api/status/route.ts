import { NextResponse } from "next/server"

export const dynamic = "force-dynamic"

export async function GET() {
  const base =
    process.env.QUANTSIGHT_API_BASE_URL ||
    "http://127.0.0.1:8765"

  try {
    const res = await fetch(`${base}/api/status`, {
      cache: "no-store",
    })

    if (!res.ok) {
      return NextResponse.json(
        {
          ok: false,
          error: `Backend HTTP ${res.status}`,
        },
        { status: 502 }
      )
    }

    const data = await res.json()

    return NextResponse.json({
      ok: true,
      data,
    })
  } catch (err) {
    return NextResponse.json(
      {
        ok: false,
        error:
          err instanceof Error
            ? err.message
            : "Backend unavailable",
      },
      { status: 502 }
    )
  }
}
