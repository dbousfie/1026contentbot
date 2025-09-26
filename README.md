# Course content bot (OpenAI + Deno + RAG)

Minimal bot that answers course/assignment questions using **your transcripts + syllabus** as context.  
Embeds in Brightspace and can optionally log to Qualtrics.

## Features
- Accepts free-text student questions
- Retrieves relevant transcript chunks from your lecture set (`/ingest`)
- Uses OpenAI to generate answers **grounded in course materials**
- Appends `Sources:` line with lecture titles (no transcript text exposed)
- Strict mode: refuses to answer if the info isn’t in your course materials
- Optionally logs `{queryText, responseText}` to Qualtrics
- Works as a standalone web page or Brightspace embed

## 1. Create your copy
- Use this template on GitHub (e.g., `course-bot-1026`)
- Make sure `main.ts`, `index.html`, and `brightspace.html` are included

## 2. Add syllabus + transcripts
- Edit `syllabus.md` with your policies or grading criteria
- Place lecture transcripts (`.txt`) in a folder (e.g. `1026t/`)
- Run the ingest script to load them into Deno KV:

```sh
deno run -A ingest_1026t.ts --token=YOUR_ADMIN_TOKEN
```

You should see progress like `Ingested 8/98 … Done.`

## 3. Deploy backend to Deno
- Sign in at https://dash.deno.com → **+ New Project** → **Import from GitHub**
- Entry point: `main.ts`
- Production branch: `main`
- Create the project (you’ll get a `https://<name>.deno.dev` URL)

## 4. Add environment variables
In **Deno → Settings → Environment Variables**, add:

```text
OPENAI_API_KEY=sk-your-openai-key
SYLLABUS_LINK=https://brightspace.university.edu/course/syllabus
QUALTRICS_API_TOKEN=(optional)
QUALTRICS_SURVEY_ID=(optional)
QUALTRICS_DATACENTER=(optional, e.g., uwo.eu)
OPENAI_MODEL=(optional, default gpt-4o-mini)
ADMIN_TOKEN=your-secret-token
STRICT_RAG=true            # restrict to transcripts + syllabus only
RAG_MIN_SCORE=0.28         # similarity threshold
RAG_TOP_K=3                # number of chunks retrieved
```

## 5. Point the frontend to your backend
In `index.html` (or `brightspace.html`), replace the fetch URL with your Deno URL:

```js
fetch("https://your-app-name.deno.dev/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: userQuery })
});
```

⚠️ Make sure you hit `/chat` (not just `/`) to use transcript RAG.

## 6. Host the frontend (GitHub Pages)
- Repo → **Settings → Pages**
- Branch: `main`, Folder: `/ (root)` → **Save**
- Use the published URL (e.g., `https://yourusername.github.io/course-bot/`)
- For Brightspace, paste `brightspace.html` into a content item or widget

## Notes
- CORS headers are included, so Brightspace iframes can call your backend.
- Responses always end with a `Sources:` line listing lecture titles.
- If strict mode is on and no match is found, the bot politely refuses.
- Responses are capped at **1500 tokens** (edit `max_tokens` in `main.ts` if needed).
- If you hit OpenAI quota/limits, switch to a cheaper model via `OPENAI_MODEL`.

## Qualtrics (optional)
- In your survey, add embedded data fields: `responseText`, `queryText`.
- Responses include an HTML comment like `<!-- Qualtrics status: 200 -->` for logging confirmation.

## Files
- `index.html` — student-facing interface
- `brightspace.html` — LMS wrapper
- `main.ts` — Deno backend (RAG + OpenAI + Qualtrics)
- `syllabus.md` — syllabus text
- `ingest_*.ts` — script to load transcripts into KV
- `README.md` — this file

## License
© Dan Bousfield. CC BY 4.0 — https://creativecommons.org/licenses/by/4.0/
