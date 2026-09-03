import { NextResponse } from "next/server";

// The AppUIModel (mode / inspector / session_history / title / subtitle) is
// injected at deploy time as JSON in DAO_AI_UI_CONFIG. Empty => client defaults.
export const dynamic = "force-dynamic";

export function GET() {
  const raw = process.env.DAO_AI_UI_CONFIG;
  try {
    return NextResponse.json(raw ? JSON.parse(raw) : {});
  } catch {
    return NextResponse.json({});
  }
}
