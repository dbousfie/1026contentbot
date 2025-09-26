# Course Bot (OpenAI + Deno + RAG)

Answers student questions using **your transcripts + syllabus** as context. Embeds in Brightspace and can optionally log to Qualtrics.

## Features
- Retrieval-augmented generation (RAG) over your **lecture transcripts**
- Strict mode (default): refuses when info isn’t in course materials
- Appends `Sources:` with **lecture titles only** (no transcript text)
- Qualtrics logging (optional)
- Admin endpoints to **ingest / retitle / wipe / stats**

---

## Quick Start

### 1) Deploy to Deno
- Dash → **+ New Project** → Import repo → Entry point: `main.ts`
- Production branch: `main` → **Create** → you get `https://<name>.deno.dev`

### 2) Environment Variables
Set in **Deno → Settings → Environment Variables**:

```
OPENAI_API_KEY=sk-...
SYLLABUS_LINK=https://<your syllabus URL>

# Model knobs
OPENAI_MODEL=gpt-4o-mini                 # optional
EMBEDDING_MODEL=text-embedding-3-small   # optional

# RAG knobs
STRICT_RAG=true         # lock responses to transcripts + syllabus
RAG_MIN_SCORE=0.20      # similarity threshold (0.18–0.30 typical)
RAG_TOP_K=5             # number of chunks retrieved

# Admin
ADMIN_TOKEN=choose-a-long-secret
# (Optional) Qualtrics
QUALTRICS_API_TOKEN=...
QUALTRICS_SURVEY_ID=...
QUALTRICS_DATACENTER=...
```

**Save & Deploy.**

---

## Endpoints

All routes are **POST**.
- `/chat` — student Q&A (also mapped at `/`)
- `/ingest` — admin-only, upload transcripts
- `/retitle` — admin-only, rename a lecture
- `/wipe` — admin-only, delete all stored lecture data (KV)
- `/stats` — admin-only, counts only (no content)

### Admin auth (hardened)
Supply either header; **whitespace is trimmed**:
```
Authorization: Bearer <ADMIN_TOKEN>
# or
X-Admin-Token: <ADMIN_TOKEN>
```

---

## Ingest

### A) 1026 (TXT files)
Run locally from the folder containing your `1026t/` directory:

```powershell
deno run -A .\ingest_1026t.ts --token=<ADMIN_TOKEN>
```

### B) 3510 (PDFs + TXTs)
Use the patched script that extracts text from PDFs and skips empty/scanned files:

```powershell
deno run -A .\ingest_pdfs_3510_patched3.ts --token=<ADMIN_TOKEN> --dir=.é0
```
If a few PDFs are skipped (0 chars), OCR them or convert to `.txt` and rerun.

---

## Wipe + Re-ingest (when you remove/rename lectures)

### Wipe
```powershell
$t = "<ADMIN_TOKEN>".Trim()
Invoke-RestMethod -Method POST `
  -Uri "https://<name>.deno.dev/wipe" `
  -Headers @{ Authorization = "Bearer $t" }
# -> "wiped N keys"
```

### (Optional) Stats
```powershell
Invoke-RestMethod -Method POST `
  -Uri "https://<name>.deno.dev/stats" `
  -Headers @{ Authorization = "Bearer $t" }
# -> {"lectures":X,"chunks":Y,"vecs":Y,"sample":[...]}
```

### Re-ingest
```powershell
# TXT
deno run -A .\ingest_1026t.ts --token=$t
# or PDF
deno run -A .\ingest_pdfs_3510_patched3.ts --token=$t --dir=.é0
```

---

## Frontend (GitHub Pages or Brightspace)

Point your `index.html` (or `brightspace.html`) to **/chat**:

```js
fetch("https://<name>.deno.dev/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: userQuery })
});
```

- Responses end with `Sources: <Lecture A>; <Lecture B>`.
- With `STRICT_RAG=true`, if nothing matches the corpus, the bot will say it can’t find that in the course materials.

---

## Qualtrics (optional)
- Add embedded data fields: `responseText`, `queryText` in your survey.
- Responses include an HTML comment like `<!-- Qualtrics status: 200 -->` for confirmation.

---

## Files
- `main.ts` — Deno backend (RAG + admin routes + Qualtrics)
- `index.html` — student UI
- `brightspace.html` — LMS wrapper
- `syllabus.md` — syllabus text
- `ingest_1026t.ts` — ingest TXT
- `ingest_pdfs_3510_patched3.ts` — ingest PDF/TXT with skips
- `README.md` — this file

---

## Troubleshooting
- **401 unauthorized** on admin routes: token mismatch. Ensure `ADMIN_TOKEN` in Deno **exactly** matches what you send; use `X-Admin-Token` or `Authorization: Bearer` (no brackets/spaces).
- **“I don’t have that information”**: lower `RAG_MIN_SCORE` (e.g., 0.20) and/or increase `RAG_TOP_K` (e.g., 5). Re-ingest if you changed files.
- **PDF extractor warnings**: noisy but harmless. Skipped files show `(pdf, 0 chars)` → OCR/convert to `.txt` and rerun.
- **Old titles show up**: run `/wipe` then re-ingest.

---

## License
© Dan Bousfield. CC BY 4.0 — https://creativecommons.org/licenses/by/4.0/
