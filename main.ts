// main.ts
// - Keeps your existing Qualtrics + syllabus flow
// - Adds private RAG via Deno KV
// - Two new routes: POST /ingest (admin-only), POST /chat (public from your front end)

import { serve } from "https://deno.land/std@0.224.0/http/server.ts";

// ---- ENV ----
const OPENAI_API_KEY = Deno.env.get("OPENAI_API_KEY");
const QUALTRICS_API_TOKEN = Deno.env.get("QUALTRICS_API_TOKEN");
const QUALTRICS_SURVEY_ID = Deno.env.get("QUALTRICS_SURVEY_ID");
const QUALTRICS_DATACENTER = Deno.env.get("QUALTRICS_DATACENTER");
const SYLLABUS_LINK = Deno.env.get("SYLLABUS_LINK") || "";
const OPENAI_MODEL = Deno.env.get("OPENAI_MODEL") || "gpt-4o-mini";           // chat model
const EMBEDDING_MODEL = Deno.env.get("EMBEDDING_MODEL") || "text-embedding-3-small";
const ADMIN_TOKEN = Deno.env.get("ADMIN_TOKEN") || "";                        // for /ingest
const kv = await Deno.openKv();

// ---- CORS helpers ----
const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
};

function corsResponse(body: string, status = 200, contentType = "text/plain") {
  return new Response(body, { status, headers: { ...CORS_HEADERS, "Content-Type": contentType } });
}

// ---- Small utilities ----
function chunkByChars(s: string, max = 1700, overlap = 200) {
  const out: string[] = [];
  for (let i = 0; i < s.length; i += max - overlap) out.push(s.slice(i, i + max));
  return out;
}

function cosine(a: number[], b: number[]) {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return d / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}

async function embedText(text: string): Promise<number[]> {
  const r = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${OPENAI_API_KEY}` },
    body: JSON.stringify({ model: EMBEDDING_MODEL, input: text }),
  });
  if (!r.ok) throw new Error(`Embedding error ${r.status}: ${await r.text()}`);
  const j = await r.json();
  return j.data[0].embedding as number[];
}

// ---- RAG: INGEST (admin-only) ----
// Body: { items: [{ id: string, title: string, text: string }, ...] }
async function handleIngest(req: Request): Promise<Response> {
  if (!OPENAI_API_KEY) return corsResponse("Missing OpenAI API key", 500);
  if (req.headers.get("authorization") !== `Bearer ${ADMIN_TOKEN}`) {
    return corsResponse("unauthorized", 401);
  }
  let payload: { items: { id: string; title: string; text: string }[] };
  try {
    payload = await req.json();
  } catch {
    return corsResponse("Invalid JSON", 400);
  }
  const items = payload.items || [];
  for (const it of items) {
    const parts = chunkByChars(it.text);
    await kv.set(["lec", it.id, "meta"], { title: it.title, n: parts.length });
    for (let i = 0; i < parts.length; i++) {
      const text = parts[i];
      const e = await embedText(text);
      await kv.set(["lec", it.id, "chunk", i], { text });  // each value well under 64 KiB
      await kv.set(["lec", it.id, "vec", i], { e });
    }
  }
  return corsResponse("ok");
}

// ---- RAG: CHAT (front end) ----
// Body: { query: string }
async function handleChat(req: Request): Promise<Response> {
  if (!OPENAI_API_KEY) return corsResponse("Missing OpenAI API key", 500);

  // Parse body
  let body: { query?: string } = {};
  try {
    body = await req.json();
  } catch {
    return corsResponse("Invalid JSON", 400);
  }
  const userQuery = (body.query || "").trim();
  if (!userQuery) return corsResponse("Missing 'query' in body", 400);

  // Load syllabus (same as your existing flow; if missing, keep going)
  const syllabus = await Deno.readTextFile("syllabus.md").catch(() => "Error loading syllabus.");

  // --- Retrieval from KV (if any vectors are present) ---
  let contextChunks: string[] = [];
  let sourceTitles: string[] = [];
  try {
    const qv = await embedText(userQuery);

    // Scan all vectors (fine for your course size; can index later if needed)
    const hits: { score: number; id: string; i: number }[] = [];
    for await (const entry of kv.list<{ e: number[] }>({ prefix: ["lec"] })) {
      // key shape: ["lec", <id>, "vec"|"chunk"|"meta", i?]
      const key = entry.key as Deno.KvKey;
      if (key.length === 4 && key[0] === "lec" && key[2] === "vec") {
        const id = String(key[1]);
        const i = Number(key[3]);
        const score = cosine(qv, entry.value.e);
        hits.push({ score, id, i });
      }
    }
    hits.sort((a, b) => b.score - a.score);
    const TOP_K = 3;
    const top = hits.slice(0, TOP_K);

    // Gather chunk texts and titles (titles only for citations)
    const titlesSet = new Set<string>();
    for (const h of top) {
      const ch = await kv.get<{ text: string }>(["lec", h.id, "chunk", h.i]);
      if (ch.value?.text) contextChunks.push(ch.value.text);
      const meta = await kv.get<{ title: string }>(["lec", h.id, "meta"]);
      if (meta.value?.title) titlesSet.add(meta.value.title);
    }
    sourceTitles = Array.from(titlesSet);
  } catch {
    // If retrieval fails for any reason, fall back to syllabus-only
    contextChunks = [];
    sourceTitles = [];
  }

  // Build the messages
  const ragContext = contextChunks.length
    ? contextChunks.map((t, i) => `(${i + 1}) ${t}`).join("\n\n---\n\n")
    : "";

  const messages = [
    {
      role: "system",
      content:
        "You are an accurate assistant. Use provided CONTEXT when available. Do not mention transcripts or retrieval. Cite lecture titles only.",
    },
    { role: "system", content: `Here is important context from syllabus.md:\n${syllabus}` },
    {
      role: "system",
      content:
        "When you rely on course material, end with a line: 'Sources: <Lecture Title 1>; <Lecture Title 2>; ...'",
    },
    {
      role: "user",
      content:
        ragContext
          ? `QUESTION:\n${userQuery}\n\nCONTEXT:\n${ragContext}`
          : userQuery,
    },
  ];

  // Call OpenAI Chat
  const openaiResponse = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: OPENAI_MODEL,
      messages,
      temperature: 0.2,
      max_tokens: 1500,
    }),
  });

  const openaiJson = await openaiResponse.json();
  const baseResponse =
    openaiJson?.choices?.[0]?.message?.content || "No response from OpenAI";

  // Ensure Sources line (titles only) if we had matches and the model forgot
  let reply = baseResponse;
  if (sourceTitles.length && !/^\s*Sources:/im.test(baseResponse)) {
    reply += `\n\nSources: ${sourceTitles.join("; ")}`;
  }

  // Append your syllabus note (kept from your original file)
  const result = `${reply}\n\nThere may be errors in my responses; always refer to the course web page: ${SYLLABUS_LINK}`;

  // Qualtrics logging (kept)
  let qualtricsStatus = "Qualtrics not called";
  if (QUALTRICS_API_TOKEN && QUALTRICS_SURVEY_ID && QUALTRICS_DATACENTER) {
    const qualtricsPayload = {
      values: { responseText: result, queryText: userQuery },
    };
    try {
      const qt = await fetch(
        `https://${QUALTRICS_DATACENTER}.qualtrics.com/API/v3/surveys/${QUALTRICS_SURVEY_ID}/responses`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-API-TOKEN": QUALTRICS_API_TOKEN,
          },
          body: JSON.stringify(qualtricsPayload),
        },
      );
      qualtricsStatus = `Qualtrics status: ${qt.status}`;
    } catch (e) {
      qualtricsStatus = `Qualtrics error: ${(e as Error).message}`;
    }
  }

  return corsResponse(`${result}\n<!-- ${qualtricsStatus} -->`);
}

// ---- Legacy handler (kept): POST / behaves like /chat ----
async function handleRoot(req: Request): Promise<Response> {
  // Keep backward compatibility for your existing front end that POSTs to "/"
  return handleChat(req);
}

// ---- Router ----
serve(async (req: Request): Promise<Response> => {
  if (req.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: CORS_HEADERS });
  }
  if (req.method !== "POST") {
    return corsResponse("Method Not Allowed", 405);
  }

  const path = new URL(req.url).pathname;
  if (path === "/ingest") return handleIngest(req);
  if (path === "/chat") return handleChat(req);
  if (path === "/") return handleRoot(req); // legacy

  // default: also treat unknown POSTs as /chat for safety
  return handleChat(req);
});
