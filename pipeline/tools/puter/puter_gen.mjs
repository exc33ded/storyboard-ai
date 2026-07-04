// One-time Puter login helper: prints an auth token for pipeline/.env.
// Usage: node puter_gen.mjs login
// (Image generation itself is done in Python — tools/image_gen.py — via the
// same POST /drivers/call the puter.js SDK makes.)
import http from "http";
import { exec } from "child_process";

const [cmd] = process.argv.slice(2);

if (cmd !== "login") {
  console.error("Usage: node puter_gen.mjs login");
  process.exit(1);
}

const token = await new Promise((resolve) => {
  const server = http.createServer((req, res) => {
    res.writeHead(200, { "Content-Type": "text/html" });
    res.end("<h2>Authentication successful — you can close this tab.</h2>");
    resolve(new URL(req.url, "http://localhost/").searchParams.get("token"));
  });
  server.listen(0, function () {
    const url = `https://puter.com/?action=authme&redirectURL=${encodeURIComponent("http://localhost:" + this.address().port)}`;
    console.error("Open this URL in your browser to log in:\n" + url);
    exec(`start "" "${url}"`);
  });
});
console.log(token);
process.exit(0);
