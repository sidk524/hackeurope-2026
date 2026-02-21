"use server";

import Anthropic from "@anthropic-ai/sdk";

const MODEL = "claude-opus-4-6";
const MAX_TOKENS = 1000;

function getAnthropicClient(): Anthropic {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    throw new Error("ANTHROPIC_API_KEY is not set");
  }
  return new Anthropic({ apiKey });
}

function extractText(content: Anthropic.Messages.ContentBlock[]): string {
  return content
    .map((block) => (block.type === "text" ? block.text : ""))
    .join("")
    .trim();
}

export async function sendClaudePrompt(prompt: string) {
  if (!prompt || !prompt.trim()) {
    throw new Error("Prompt is required");
  }

  const anthropic = getAnthropicClient();
  const message = await anthropic.messages.create({
    model: MODEL,
    max_tokens: MAX_TOKENS,
    messages: [{ role: "user", content: prompt }],
  });

  return {
    id: message.id,
    model: message.model,
    text: extractText(message.content),
    stopReason: message.stop_reason ?? null,
    usage: message.usage ?? null,
  };
}
