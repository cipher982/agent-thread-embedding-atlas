import { createServer } from "node:http";
import { createReadStream, existsSync } from "node:fs";
import { extname, join, normalize } from "node:path";

const PORT = Number(process.env.PORT || 80);
const ROOT = join(process.cwd(), "dist");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".ico": "image/x-icon"
};

function sendFile(res, filePath) {
  const ext = extname(filePath).toLowerCase();
  const contentType = MIME[ext] || "application/octet-stream";
  res.writeHead(200, { "Content-Type": contentType, "Cache-Control": "public, max-age=300" });
  createReadStream(filePath).pipe(res);
}

createServer((req, res) => {
  const rawPath = (req.url || "/").split("?")[0];

  if (rawPath === "/health") {
    res.writeHead(200, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("ok");
    return;
  }

  const safePath = normalize(rawPath).replace(/^\.\.(\/|\\|$)/, "");
  const requested = join(ROOT, safePath);

  if (existsSync(requested) && !rawPath.endsWith("/")) {
    sendFile(res, requested);
    return;
  }

  if (existsSync(join(requested, "index.html"))) {
    sendFile(res, join(requested, "index.html"));
    return;
  }

  const spaIndex = join(ROOT, "index.html");
  if (existsSync(spaIndex)) {
    sendFile(res, spaIndex);
    return;
  }

  res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
  res.end("Not found");
}).listen(PORT, () => {
  console.log(`embed atlas serving on :${PORT}`);
});
