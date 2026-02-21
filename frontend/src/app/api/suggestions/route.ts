import { NextResponse } from "next/server";

type SuggestionPayload = {
  suggestion?: string;
  token?: string;
  projectName?: string;
};

export async function POST(request: Request) {
  let payload: SuggestionPayload = {};

  try {
    payload = (await request.json()) as SuggestionPayload;
  } catch {
    return NextResponse.json(
      { ok: false, error: "Invalid JSON payload" },
      { status: 400 },
    );
  }

  const { suggestion, token, projectName } = payload;

  if (!suggestion || !token || !projectName) {
    return NextResponse.json(
      { ok: false, error: "Missing suggestion, token, or projectName" },
      { status: 400 },
    );
  }

  console.log("Pipeline suggestion received", {
    suggestion,
    token,
    projectName,
  });

  return NextResponse.json({ ok: true });
}
