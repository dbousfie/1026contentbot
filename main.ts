// main.ts — strict RAG + Qualtrics + syllabus
// Routes:
//   POST /ingest   (admin-only) -> upload transcripts [{ id, title, text }]
//   POST /retitle  (admin-only) -> update a lecture title { id, title }
//   POST /wipe     (admin-only) -> delete ALL stored lecture data
//   POST /stats    (admin-only) -> counts only (no content): {lectures,chunks,vecs,sample}
//   POST /chat     (front-end)  -> RAG answer, cites lecture titles only
//   POST /         (legacy)     -> same as /chat

import { serve } from "https://deno.land/std@0.224.0/http/server.ts";

// ---------- ENV ----------
const OPENAI_API_KEY       = Deno.env.get("OPENAI_API_KEY");
const OPENAI_MODEL         = Deno.env.get("OPENAI_MODEL") || "gpt-4o-mini";
const EMBEDDING_MODEL      = Deno.env.get("EMBEDDING_MODEL") || "text-embedding-3-small";
const ADMIN_TOKEN          = (Deno.env.get("ADMIN_TOKEN") || "").trim();

const QUALTRICS_API_TOKEN  = Deno.env.get("QUALTRICS_API_TOKEN");
const QUALTRICS_SURVEY_ID  = Deno.env.get("QUALTRICS_SURVEY_ID");
const QUALTRICS_DATACENTER = Deno.env.get("QUALTRICS_DATACENTER");
const SYLLABUS_LINK        = Deno.env.get("SYLLABUS_LINK") || "";

// RAG knobs
const STRICT_RAG = (Deno.env.get("STRICT_RAG") ?? "true").toLowerCase() === "true";
const MIN_SCORE  = Number(Deno.env.get("RAG_MIN_SCORE") ?? "0.28");
const TOP_K      = Number(Deno.env.get("RAG_TOP_K") ?? "3");

const kv = await Deno.openKv();

// ---------- CORS ----------
const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Admin-Token",
};
const cors = (body: string, status = 200, ct = "text/plain") =>
  new Response(body, { status, headers: { ...CORS_HEADERS, "Content-Type": ct } });

// ---------- AUTH (whitespace- & header-safe) ----------
function adminTokenFrom(req: Request): string {
  const auth = req.headers.get("authorization") || "";
  const alt  = req.headers.get("x-admin-token") || "";
  let bearer = "";
  if (/^Bearer\b/i.test(auth)) bearer = auth.replace(/^Bearer\s+/i, "");
  return (bearer || alt).trim();
}
function requireAdmin(req: Request): boolean {
  if (!ADMIN_TOKEN) return false;
  return adminTokenFrom(req) === ADMIN_TOKEN;
}

// ---------- UTILS ----------
function chunkByChars(s: string, max = 1700, overlap = 200) {
  const out: string[] = [];
  for (let i = 0; i < s.length; i += max - overlap) out.push(s.slice(i, i + max));
  return out;
}
function cosine(a: number[], b: number[]) {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
  return d / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}
async function embedText(text: string): Promise<number[]> {
  if (!OPENAI_API_KEY) throw new Error("Missing OpenAI API key");
  const r = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${OPENAI_API_KEY}` },
    body: JSON.stringify({ model: EMBEDDING_MODEL, input: text }),
  });
  if (!r.ok) throw new Error(`Embedding error ${r.status}: ${await r.text()}`);
  const j = await r.json();
  return j.data[0].embedding as number[];
}
function safeFooter(): string {
  return /^https?:\/\//i.test(SYLLABUS_LINK)
    ? `\n\nThere may be errors in my responses; always refer to the course web page: ${SYLLABUS_LINK}`
    : `\n\nThere may be errors in my responses; consult the official course page.`;
}

// ---------- /ingest (admin-only) ----------
async function handleIngest(req: Request): Promise<Response> {
  if (!requireAdmin(req)) return cors("unauthorized", 401);
  let payload: { items: { id: string; title: string; text: string }[] };
  try { payload = await req.json(); } catch { return cors("Invalid JSON", 400); }
  const items = payload.items || [];
  for (const it of items) {
    const parts = chunkByChars(it.text);
    await kv.set(["lec", it.id, "meta"], { title: it.title, n: parts.length });
    for (let i = 0; i < parts.length; i++) {
      const text = parts[i];
      const e = await embedText(text);
      await kv.set(["lec", it.id, "chunk", i], { text }); // <= 64KiB each
      await kv.set(["lec", it.id, "vec",   i], { e });
    }
  }
  return cors("ok");
}

// ---------- /retitle (admin-only) ----------
async function handleRetitle(req: Request): Promise<Response> {
  if (!requireAdmin(req)) return cors("unauthorized", 401);
  let body: { id?: string; title?: string } = {};
  try { body = await req.json(); } catch { return cors("Invalid JSON", 400); }
  if (!body.id || !body.title) return cors("id and title required", 400);
  const meta = await kv.get<{ title: string; n: number }>(["lec", body.id, "meta"]);
  if (!meta.value) return cors("not found", 404);
  await kv.set(["lec", body.id, "meta"], { ...meta.value, title: body.title });
  return cors("ok");
}

// ---------- /wipe (admin-only) ----------
async function handleWipe(req: Request): Promise<Response> {
  if (!requireAdmin(req)) return cors("unauthorized", 401);
  let deleted = 0;
  for await (const e of kv.list({ prefix: ["lec"] })) {
    await kv.delete(e.key);
    deleted++;
  }
  return cors(`wiped ${deleted} keys`);
}

// ---------- /stats (admin-only) ----------
async function handleStats(req: Request): Promise<Response> {
  if (!requireAdmin(req)) return cors("unauthorized", 401);
  let chunks = 0, vecs = 0, lectures = 0;
  const titles = new Set<string>();
  for await (const e of kv.list({ prefix: ["lec"] })) {
    const k = e.key as Deno.KvKey;
    if (k.length === 3 && k[2] === "meta") { lectures++; titles.add((e.value as any)?.title ?? ""); }
    if (k.length === 4 && k[2] === "chunk") chunks++;
    if (k.length === 4 && k[2] === "vec")   vecs++;
  }
  const sample = Array.from(titles).filter(Boolean).slice(0, 10);
  return cors(JSON.stringify({ lectures, chunks, vecs, sample }, null, 2), 200, "application/json");
}

// ---------- /chat (front-end) ----------
async function handleChat(req: Request): Promise<Response> {
  if (!OPENAI_API_KEY) return cors("Missing OpenAI API key", 500);

  let body: { query?: string } = {};
  try { body = await req.json(); } catch { return cors("Invalid JSON", 400); }
  const userQuery = (body.query || "").trim();
  if (!userQuery) return cors("Missing 'query' in body", 400);

  const syllabus = await Deno.readTextFile("syllabus.md").catch(() => "Error loading syllabus.");

  type Hit = { score: number; id: string; i: number; text: string; title: string };
  const hits: Hit[] = [];
  let hadRetrievalError = false;

  try {
    const qv = await embedText(userQuery);
    for await (const entry of kv.list<{ e: number[] }>({ prefix: ["lec"] })) {
      const key = entry.key as Deno.KvKey;
      if (key.length === 4 && key[0] === "lec" && key[2] === "vec") {
        const id = String(key[1]); const i = Number(key[3]);
        const score = cosine(qv, entry.value.e);
        const ch   = await kv.get<{ text: string }>(["lec", id, "chunk", i]);
        const meta = await kv.get<{ title: string }>(["lec", id, "meta"]);
        if (!ch.value?.text || !meta.value?.title) continue;
        hits.push({ score, id, i, text: ch.value.text, title: meta.value.title });
      }
    }
  } catch { hadRetrievalError = true; }

  hits.sort((a, b) => b.score - a.score);

  const filtered = hits.filter(h => h.score >= MIN_SCORE).slice(0, TOP_K);
  if (STRICT_RAG && !filtered.length && !hadRetrievalError) {
    return cors(`I can’t find this in the course materials I have. Please check lecture titles or rephrase.${safeFooter()}`);
  }
  const top = filtered.length ? filtered : hits.slice(0, TOP_K);

  const context = top.length
    ? top.map((h, idx) => `(${idx + 1}) ${h.title}\n${h.text}`).join("\n\n---\n\n")
    : "";

  const sourceTitles = Array.from(new Set(top.map(h => h.title)));

  const messages = [
    {
      role: "system" as const,
      content: STRICT_RAG
        ? "Answer ONLY using the CONTEXT and the syllabus text provided. If the answer is not in the CONTEXT/syllabus, say you don’t have that information. Do not speculate. Do not mention transcripts or retrieval."
        : "Use the CONTEXT and syllabus when provided. Prefer them over prior knowledge. Do not mention transcripts or retrieval."
    },
    { role: "system" as const, content: `Here is important context from syllabus.md:\n${syllabus}` },
    { role: "system" as const, content: "After your answer, add a line starting with 'Sources:' followed by the lecture titles you used, separated by semicolons. Cite titles ONLY." },
    { role: "user"   as const, content: context ? `QUESTION:\n${userQuery}\n\nCONTEXT:\n${context}` : userQuery },
  ];

  const r = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${OPENAI_API_KEY}` },
    body: JSON.stringify({ model: OPENAI_MODEL, messages, temperature: 0.2, max_tokens: 1500 }),
  });
  const j = await r.json();
  const base = j?.choices?.[0]?.message?.content || "No response from OpenAI";

  let reply = base;
  if (sourceTitles.length && !/^\s*Sources:/im.test(base)) {
    reply += `\n\nSources: ${sourceTitles.join("; ")}`;
  }

  let qualtricsStatus = "Qualtrics not called";
  if (QUALTRICS_API_TOKEN && QUALTRICS_SURVEY_ID && QUALTRICS_DATACENTER) {
    try {
      const qt = await fetch(
        `https://${QUALTRICS_DATACENTER}.qualtrics.com/API/v3/surveys/${QUALTRICS_SURVEY_ID}/responses`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-API-TOKEN": QUALTRICS_API_TOKEN },
          body: JSON.stringify({ values: { responseText: reply + safeFooter(), queryText: userQuery } }),
        },
      );
      qualtricsStatus = `Qualtrics status: ${qt.status}`;
    } catch (e) {
      qualtricsStatus = `Qualtrics error: ${(e as Error).message}`;
    }
  }

  return cors(`${reply}${safeFooter()}\n<!-- ${qualtricsStatus} -->`);
}

// ---------- legacy root ----------
const handleRoot = handleChat;

// ---------- Router ----------
serve(async (req: Request): Promise<Response> => {
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS_HEADERS });
  if (req.method !== "POST")   return cors("Method Not Allowed", 405);

  const path = new URL(req.url).pathname;
  if (path === "/ingest")  return handleIngest(req);
  if (path === "/retitle") return handleRetitle(req);
  if (path === "/wipe")    return handleWipe(req);
  if (path === "/stats")   return handleStats(req);
  if (path === "/chat")    return handleChat(req);
  if (path === "/")        return handleRoot(req);
  return handleChat(req);
});
