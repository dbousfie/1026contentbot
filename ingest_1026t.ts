// ingest_1026t.ts
// Usage:
// deno run -A ingest_1026t.ts --token YOUR_ADMIN_TOKEN

const args = new Map(Deno.args.map(a => a.split("=")));
const token = args.get("--token");
if (!token) {
  console.error("Usage: deno run -A ingest_1026t.ts --token YOUR_ADMIN_TOKEN");
  Deno.exit(1);
}

const server = "https://1026contentbot.deno.dev"; // your Deno deploy URL

// make a safe id from file name
function slug(s: string) {
  return s.toLowerCase()
    .replace(/\.[^.]+$/, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");
}

const entries: {id: string; title: string; text: string}[] = [];
for await (const f of Deno.readDir("./1026t")) {
  if (!f.isFile || !f.name.toLowerCase().endsWith(".txt")) continue;
  const path = `./1026t/${f.name}`;
  const text = await Deno.readTextFile(path);
  const title = f.name.replace(/\.txt$/i, "");
  const id = slug(title);
  entries.push({ id, title, text });
}

if (!entries.length) {
  console.error("No .txt files found in ./1026t");
  Deno.exit(1);
}

const BATCH = 8; // upload in safe chunks
for (let i = 0; i < entries.length; i += BATCH) {
  const items = entries.slice(i, i + BATCH);
  const res = await fetch(`${server}/ingest`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${token}`
    },
    body: JSON.stringify({ items }),
  });
  if (!res.ok) {
    console.error(`Ingest failed at batch starting ${i}: ${res.status} ${await res.text()}`);
    Deno.exit(1);
  }
  console.log(`Ingested ${i + items.length}/${entries.length}`);
}
console.log("Done.");
