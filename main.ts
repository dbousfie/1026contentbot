// main.ts — drop-in server for 1026contentbot (Bot 1)
// - KV schema: ["lec", slug, "meta" | "chunk" | "vec", i]
// - Endpoints:
//   POST /ingest   (admin)  -> {ok:true,count}
//   POST /wipe     (admin)  -> {wiped}
//   POST /stats    (admin)  -> {lectures,chunks,vecs,sample:[...]}
//   POST /         (public) -> RAG answer text
//
// Env:
//   ADMIN_TOKEN (required for admin routes)
//   OPENAI_API_KEY (required for embeddings & chat)
//   EMBEDDING_MODEL or EMBED_MODEL (optional, default text-embedding-3-small)
//   OPENAI_MODEL (optional, default gpt-4o-mini)
//   RAG_TOP_K (optional, default 3)
//   RAG_MIN_SCORE (optional, default 0.28)

const kv = await Deno.openKv();

const ADMIN_TOKEN = Deno.env.get("ADMIN_TOKEN") ?? "";
const OPENAI_API_KEY = Deno.env.get("OPENAI_API_KEY") ?? "";
const EMBED_MODEL =
  Deno.env.get("EMBEDDING_MODEL") ??
  Deno.env.get("EMBED_MODEL") ??
  "text-embedding-3-small";
const CHAT_MODEL = Deno.env.get("OPENAI_MODEL") ?? "gpt-4o-mini";
const RAG_TOP_K = Number(Deno.env.get("RAG_TOP_K") ?? "3");
const RAG_MIN_SCORE = Number(Deno.env.get("RAG_MIN_SCORE") ?? "0.28");

function cors(h: Headers) {
  h.set("Access-Control-Allow-Origin", "*");
  h.set("Access-Control-Allow-Headers", "*");
  h.set("Access-Control-Allow-Methods", "POST,OPTIONS");
  return h;
}
function isAdmin(req: Request) {
  const h = req.headers;
  const tok =
    h.get("X-Admin-Token") ||
    (h.get("Authorization")?.startsWith("Bearer ")
      ? h.get("Authorization")!.slice(7)
      : "");
  return tok && tok === ADMIN_TOKEN;
}
function bad(status: number, msg: string) {
  return new Response(msg, {
    status,
    headers: cors(new Headers({ "content-type": "text/plain" })),
  });
}

type Item = { id: string; title: string; text: string };

function slugify(s: string) {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
}

// simple chunker ~1000 chars
function chunkText(t: string, size = 1000): string[] {
  const out: string[] = [];
  let i = 0;
  while (i < t.length) {
    out.push(t.slice(i, i + size));
    i += size;
  }
  return out;
}

async function embedBatch(texts: string[]): Promise<number[][]> {
  if (!OPENAI_API_KEY) throw new Error("missing OPENAI_API_KEY");
  const res = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      Authorization: `Bearer ${OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: EMBED_MODEL,
      input: texts,
    }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`embed ${res.status}: ${err}`);
  }
  const j = await res.json();
  return j.data.map((d: any) => d.embedding as number[]);
}

function cosine(a: number[], b: number[]) {
  let dot = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}

async function handleIngest(req: Request) {
  if (!isAdmin(req)) return bad(401, "unauthorized");
  const { items } = (await req.json()) as { items: Item[] };
  if (!Array.isArray(items)) return bad(400, "invalid payload");
  let wrote = 0;

  for (const it of items) {
    const id = slugify(it.id || it.title);
    const chunks = chunkText(it.text);
    // embed in batches of 16
    const batchSize = 16;
    const vecs: number[][] = [];
    for (let i = 0; i < chunks.length; i += batchSize) {
      const slice = chunks.slice(i, i + batchSize);
      const e = await embedBatch(slice);
      vecs.push(...e);
    }

    const meta = { title: it.title, n: chunks.length, model: EMBED_MODEL };
    await kv.set(["lec", id, "meta"], meta);
    for (let i = 0; i < chunks.length; i++) {
      await kv.set(["lec", id, "chunk", i], { text: chunks[i] });
      await kv.set(["lec", id, "vec", i], { e: vecs[i] });
    }
    wrote++;
  }
  return new Response(JSON.stringify({ ok: true, count: wrote }), {
    headers: cors(new Headers({ "content-type": "application/json" })),
  });
}

async function handleWipe(req: Request) {
  if (!isAdmin(req)) return bad(401, "unauthorized");
  let wiped = 0;
  for await (const e of kv.list({ prefix: ["lec"] })) {
    await kv.delete(e.key);
    wiped++;
  }
  return new Response(JSON.stringify({ wiped }), {
    headers: cors(new Headers({ "content-type": "application/json" })),
  });
}

async function handleStats(req: Request) {
  if (!isAdmin(req)) return bad(401, "unauthorized");
  let lectures = 0,
    chunks = 0,
    vecs = 0;
  const titles = new Set<string>();
  for await (const e of kv.list({ prefix: ["lec"] })) {
    const k = e.key as Deno.KvKey;
    if (k.length === 3 && k[2] === "meta") {
      lectures++;
      const v = e.value as { title?: string };
      if (v?.title) titles.add(v.title);
    } else if (k.length === 4 && k[2] === "chunk") {
      chunks++;
    } else if (k.length === 4 && k[2] === "vec") {
      vecs++;
    }
  }
  return new Response(
    JSON.stringify({
      lectures,
      chunks,
      vecs,
      sample: Array.from(titles).slice(0, 6),
    }),
    { headers: cors(new Headers({ "content-type": "application/json" })) },
  );
}

async function handleChat(req: Request) {
  const { query } = (await req.json().catch(() => ({ query: "" }))) as {
    query: string;
  };
  if (!query || !OPENAI_API_KEY) {
    return bad(400, "missing query or OPENAI_API_KEY");
  }

  // 1) embed query
  const [qv] = await embedBatch([query]);

  // 2) scan vectors and score (brute-force; OK for small corpora)
  const hits: { id: string; i: number; text: string; score: number }[] = [];
  for await (const e of kv.list({ prefix: ["lec"] })) {
    const k = e.key as Deno.KvKey;
    if (k.length === 4 && k[2] === "vec") {
      const [_, id, __, i] = k as [string, string, string, number];
      const v = (e.value as { e: number[] }).e;
      const s = cosine(qv, v);
      if (s >= RAG_MIN_SCORE) {
        const ch = await kv.get(["lec", id, "chunk", i]);
        const text = (ch.value as { text: string })?.text ?? "";
        hits.push({ id, i, text, score: s });
      }
    }
  }
  hits.sort((a, b) => b.score - a.score);
  const top = hits.slice(0, RAG_TOP_K);

  const context = top
    .map(
      (h, idx) =>
        `[[${idx + 1}]] (doc=${h.id} part=${h.i} score=${h.score.toFixed(3)})\n${h.text}`,
    )
    .join("\n\n");

  const system =
    "You are a course content assistant. Answer using the provided context. If the answer is not in the context, say you don't have that content.";
  const userMsg = `Question: ${query}\n\nContext:\n${context || "(no matching context)"}`;

  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      Authorization: `Bearer ${OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: CHAT_MODEL,
      messages: [
        { role: "system", content: system },
        { role: "user", content: userMsg },
      ],
      temperature: 0.2,
    }),
  });
  if (!res.ok) {
    const err = await res.text();
    return bad(res.status, err);
  }
  const j = await res.json();
  const text = j.choices?.[0]?.message?.content ?? "";
  return new Response(text, {
    headers: cors(new Headers({ "content-type": "text/plain" })),
  });
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: cors(new Headers()) });
  }
  const url = new URL(req.url);
  const p = url.pathname;

  try {
    if (p === "/ingest" && req.method === "POST") return await handleIngest(req);
    if (p === "/wipe" && req.method === "POST") return await handleWipe(req);
    if (p === "/stats" && req.method === "POST") return await handleStats(req);
    if (p === "/" && req.method === "POST") return await handleChat(req);
    return bad(405, "Method Not Allowed");
  } catch (e) {
    return new Response(`error: ${e?.message ?? e}`, {
      status: 500,
      headers: cors(new Headers({ "content-type": "text/plain" })),
    });
  }
});
