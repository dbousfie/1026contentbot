// main.ts — KV-lean RAG with packed chunks + RAM index + exact Sources

import { serve } from "https://deno.land/std@0.224.0/http/server.ts";

/* ===== ENV ===== */
const OPENAI_API_KEY       = Deno.env.get("OPENAI_API_KEY") ?? "";
const OPENAI_MODEL         = Deno.env.get("OPENAI_MODEL") ?? "gpt-4o-mini";
const EMBEDDING_MODEL      = Deno.env.get("EMBEDDING_MODEL") ?? "text-embedding-3-small";
const ADMIN_TOKEN          = (Deno.env.get("ADMIN_TOKEN") ?? "").trim();
const SYLLABUS_LINK        = Deno.env.get("SYLLABUS_LINK") ?? "";

const STRICT_RAG = (Deno.env.get("STRICT_RAG") ?? "true").toLowerCase() === "true";
const MIN_SCORE  = Number(Deno.env.get("RAG_MIN_SCORE") ?? "0.25");
const TOP_K      = Number(Deno.env.get("RAG_TOP_K") ?? "3");
const CACHE_TTL_MIN = Number(Deno.env.get("CACHE_TTL_MIN") ?? "60"); // refresh index hourly by default

/* ===== CORS ===== */
const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Admin-Token",
};
const respond = (body: string, status = 200, ct = "text/plain") =>
  new Response(body, { status, headers: { ...CORS, "Content-Type": ct } });

/* ===== KV ===== */
const kv = await Deno.openKv();

/* ===== Types ===== */
type Pack = { id: string; i: number; title: string; text: string; e: number[] };
type Meta = { title: string; n: number };

/* ===== Admin auth (trim + alt header) ===== */
function adminTokenFrom(req: Request): string {
  const auth = req.headers.get("authorization") ?? "";
  const alt  = req.headers.get("x-admin-token") ?? "";
  const bearer = /^Bearer\s+/i.test(auth) ? auth.replace(/^Bearer\s+/i, "") : "";
  return (bearer || alt).trim();
}
function requireAdmin(req: Request) {
  return ADMIN_TOKEN && adminTokenFrom(req) === ADMIN_TOKEN;
}

/* ===== Helpers ===== */
function footer() {
  return SYLLABUS_LINK
    ? `\n\nThere may be errors in my responses; always refer to the course web page: ${SYLLABUS_LINK}`
    : `\n\nThere may be errors in my responses; consult the official course page.`;
}
function chunkByChars(s: string, max = 1700, overlap = 200) {
  const out: string[] = [];
  for (let i = 0; i < s.length; i += max - overlap) out.push(s.slice(i, i + max));
  return out;
}
function cosine(a: number[], b: number[]) {
  let d=0,na=0,nb=0;
  for (let i=0;i<a.length;i++){ d+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
  return d/(Math.sqrt(na)*Math.sqrt(nb)+1e-8);
}
async function embed(text: string): Promise<number[]> {
  const r = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: { "Content-Type":"application/json", Authorization:`Bearer ${OPENAI_API_KEY}` },
    body: JSON.stringify({ model: EMBEDDING_MODEL, input: text }),
  });
  if (!r.ok) throw new Error(`embedding ${r.status}: ${await r.text()}`);
  const j = await r.json();
  return j.data[0].embedding as number[];
}

/* ===== RAM index ===== */
let INDEX: Pack[] | null = null;
let INDEX_LOADED_AT = 0;
let INDEX_VERSION_SEEN = 0; // bump on ingest/wipe

async function loadIndex(force = false) {
  const now = Date.now();
  const ttl = CACHE_TTL_MIN * 60_000;
  const ver = (await kv.get<number>(["index_version"])).value ?? 0;

  const needReload =
    force ||
    !INDEX ||
    now - INDEX_LOADED_AT > ttl ||
    ver !== INDEX_VERSION_SEEN;

  if (!needReload) return;

  const packs: Pack[] = [];
  for await (const e of kv.list<Pack>({ prefix: ["pack"] })) {
    // keys: ["pack", id, i]
    const k = e.key as Deno.KvKey;
    if (k.length === 3 && k[0] === "pack") {
      const v = e.value as any;
      if (v?.title && v?.text && v?.e) {
        packs.push({ id: String(k[1]), i: Number(k[2]), title: v.title, text: v.text, e: v.e });
      }
    }
  }
  INDEX = packs;
  INDEX_LOADED_AT = now;
  INDEX_VERSION_SEEN = ver;
}

/* ===== Admin endpoints ===== */

// POST /ingest { items: [{id, title, text}] }
async function handleIngest(req: Request) {
  if (!requireAdmin(req)) return respond("unauthorized", 401);
  let body: { items: { id: string; title: string; text: string }[] } = { items: [] };
  try { body = await req.json(); } catch { return respond("Invalid JSON", 400); }
  if (!OPENAI_API_KEY) return respond("Missing OpenAI API key", 500);

  for (const it of body.items || []) {
    const parts = chunkByChars(it.text);
    // Write meta (1 key) + packed chunks (1 key each)
    await kv.set(["lec", it.id, "meta"], { title: it.title, n: parts.length } as Meta);
    for (let i = 0; i < parts.length; i++) {
      const text = parts[i];
      const e = await embed(text);
      await kv.set(["pack", it.id, i], { title: it.title, text, e } satisfies Omit<Pack,"id"|"i">);
    }
  }
  // bump index version so live instances reload in RAM
  const cur = (await kv.get<number>(["index_version"])).value ?? 0;
  await kv.set(["index_version"], cur + 1);
  // refresh local cache
  await loadIndex(true);

  return respond("ok");
}

// POST /retitle { id, title }
async function handleRetitle(req: Request) {
  if (!requireAdmin(req)) return respond("unauthorized", 401);
  let body: { id?: string; title?: string } = {};
  try { body = await req.json(); } catch { return respond("Invalid JSON", 400); }
  if (!body.id || !body.title) return respond("id and title required", 400);

  const meta = await kv.get<Meta>(["lec", body.id, "meta"]);
  if (!meta.value) return respond("not found", 404);

  // update meta title
  await kv.set(["lec", body.id, "meta"], { ...meta.value, title: body.title });

  // update titles inside packed chunks (no re-embed)
  for await (const e of kv.list({ prefix: ["pack", body.id] })) {
    const v = e.value as any;
    await kv.set(e.key, { ...v, title: body.title });
  }

  const cur = (await kv.get<number>(["index_version"])).value ?? 0;
  await kv.set(["index_version"], cur + 1);
  await loadIndex(true);

  return respond("ok");
}

// POST /wipe
async function handleWipe(req: Request) {
  if (!requireAdmin(req)) return respond("unauthorized", 401);
  let n = 0;
  for await (const e of kv.list({ prefix: ["lec"] })) { await kv.delete(e.key); n++; }
  for await (const e of kv.list({ prefix: ["pack"] })) { await kv.delete(e.key); n++; }
  await kv.set(["index_version"], ((await kv.get<number>(["index_version"])).value ?? 0) + 1);
  INDEX = null;
  return respond(`wiped ${n} keys`);
}

// POST /stats
async function handleStats(req: Request) {
  if (!requireAdmin(req)) return respond("unauthorized", 401);
  await loadIndex();
  const titles = new Set(INDEX?.map(p => p.title) ?? []);
  return respond(JSON.stringify({
    lectures: titles.size,
    chunks: INDEX?.length ?? 0,
    vecs: INDEX?.length ?? 0,
    sample: Array.from(titles).slice(0, 10),
  }, null, 2), 200, "application/json");
}

/* ===== Chat ===== */
// POST /chat  { query }
async function handleChat(req: Request) {
  if (!OPENAI_API_KEY) return respond("Missing OpenAI API key", 500);
  let body: { query?: string } = {};
  try { body = await req.json(); } catch { return respond("Invalid JSON", 400); }
  const userQuery = (body.query ?? "").trim();
  if (!userQuery) return respond("Missing 'query' in body", 400);

  const syllabus = await Deno.readTextFile("syllabus.md").catch(() => "Error loading syllabus.");

  await loadIndex(); // warm or reuse RAM index
  const index = INDEX ?? [];
  if (!index.length) {
    return respond(`I don’t have any course materials loaded yet. Please ingest and try again.${footer()}`);
  }

  const qv = await embed(userQuery);
  const scored = index.map(p => ({ s: cosine(qv, p.e), p }))
                      .sort((a,b)=>b.s-a.s);

  const hits = scored.filter(x => x.s >= MIN_SCORE).slice(0, TOP_K).map(x=>x.p);
  if (STRICT_RAG && hits.length === 0) {
    return respond(`I can’t find this in the course materials I have. Please check lecture titles or rephrase.${footer()}`);
  }
  const top = hits.length ? hits : scored.slice(0, TOP_K).map(x=>x.p);

  const context = top.map((h, i) => `(${i+1}) ${h.title}\n${h.text}`).join("\n\n---\n\n");
  const sourceTitles = Array.from(new Set(top.map(h => h.title)));

  const messages = [
    { role: "system", content: STRICT_RAG
        ? "Answer ONLY using the CONTEXT and the syllabus text provided. If the answer is not in the CONTEXT/syllabus, say you don’t have that information. Don’t mention retrieval."
        : "Use the CONTEXT and syllabus when provided. Prefer them. Don’t mention retrieval." },
    { role: "system", content: `Here is important context from syllabus.md:\n${syllabus}` },
    { role: "user",   content: `QUESTION:\n${userQuery}\n\nCONTEXT:\n${context}` },
  ] as const;

  const r = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type":"application/json", Authorization:`Bearer ${OPENAI_API_KEY}` },
    body: JSON.stringify({ model: OPENAI_MODEL, messages, temperature: 0.2, max_tokens: 1500 }),
  });
  const j = await r.json();
  const base = j?.choices?.[0]?.message?.content || "No response";

  // enforce exact Sources (ignore model-written ones)
  const cleaned = String(base).replace(/^\s*Sources:.*$/gmi, "").trim();
  const exactSources = `\n\nSources: ${sourceTitles.join("; ")}`;
  return respond(`${cleaned}${exactSources}${footer()}`);
}

/* ===== Router ===== */
serve(async (req: Request) => {
  const path = new URL(req.url).pathname;
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS });
  if (req.method !== "POST")   return respond("Method Not Allowed", 405);

  if (path === "/ingest")  return handleIngest(req);
  if (path === "/retitle") return handleRetitle(req);
  if (path === "/wipe")    return handleWipe(req);
  if (path === "/stats")   return handleStats(req);
  if (path === "/chat" || path === "/") return handleChat(req);
  return handleChat(req);
});
