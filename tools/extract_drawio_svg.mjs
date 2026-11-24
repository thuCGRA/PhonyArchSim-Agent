import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import process from "node:process";
import zlib from "node:zlib";

function drawioFragment(xml) {
  const encoded = encodeURIComponent(xml);
  const compressed = zlib.deflateRawSync(Buffer.from(encoded, "utf8"));
  return encodeURIComponent(compressed.toString("base64"));
}

async function delay(ms) {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

async function cdpConnect(webSocketUrl) {
  const socket = new WebSocket(webSocketUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", reject, { once: true });
  });
  let nextId = 1;
  const pending = new Map();
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const { resolve, reject } = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) reject(new Error(JSON.stringify(message.error)));
      else resolve(message.result);
    }
  });
  return {
    call(method, params = {}) {
      const id = nextId++;
      socket.send(JSON.stringify({ id, method, params }));
      return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
    },
    close() {
      socket.close();
    },
  };
}

async function main() {
  const [input, output, edge] = process.argv.slice(2);
  if (!input || !output || !edge) {
    throw new Error("Usage: node extract_drawio_svg.mjs input.drawio output.svg msedge.exe");
  }

  const xml = await readFile(input, "utf8");
  const pageWidth = Number(xml.match(/pageWidth="([0-9.]+)"/)?.[1] ?? 422);
  const pageHeight = Number(xml.match(/pageHeight="([0-9.]+)"/)?.[1] ?? 333);
  const url = `https://viewer.diagrams.net/?highlight=0000ff&layers=1&nav=0&title=figure#R${drawioFragment(xml)}`;
  const profile = await mkdtemp(path.join(tmpdir(), "codex-drawio-cdp-"));
  const port = 9300 + Math.floor(Math.random() * 500);
  const browser = spawn(
    edge,
    [
      "--headless=new",
      "--disable-gpu",
      "--disable-extensions",
      "--no-first-run",
      `--remote-debugging-port=${port}`,
      `--user-data-dir=${profile}`,
      url,
    ],
    { stdio: "ignore" },
  );

  try {
    let pages;
    for (let attempt = 0; attempt < 60; attempt += 1) {
      try {
        pages = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
        if (pages.some((page) => page.type === "page")) break;
      } catch {}
      await delay(500);
    }
    const page = pages?.find((item) => item.type === "page");
    if (!page) throw new Error("Could not connect to the Edge page through CDP");
    const cdp = await cdpConnect(page.webSocketDebuggerUrl);
    try {
      await cdp.call("Runtime.enable");
      let svg = "";
      for (let attempt = 0; attempt < 60; attempt += 1) {
        const result = await cdp.call("Runtime.evaluate", {
          returnByValue: true,
          expression: `(() => {
            const candidates = [...document.querySelectorAll('svg')]
              .filter((node) => node.querySelectorAll('*').length > 10)
              .sort((a, b) => b.outerHTML.length - a.outerHTML.length);
            const source = candidates[0];
            if (!source) return '';
            const graphGroup = [...source.querySelectorAll('g[transform*="scale"]')]
              .sort((a, b) => b.querySelectorAll('*').length - a.querySelectorAll('*').length)[0];
            if (!graphGroup || graphGroup.querySelectorAll('*').length < 10) return '';
            const clone = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
            clone.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');
            clone.setAttribute('viewBox', '0 0 ${pageWidth} ${pageHeight}');
            clone.setAttribute('width', '${pageWidth}');
            clone.setAttribute('height', '${pageHeight}');
            const content = graphGroup.cloneNode(true);
            content.removeAttribute('transform');
            clone.appendChild(content);
            return new XMLSerializer().serializeToString(clone);
          })()`,
        });
        svg = result.result?.value ?? "";
        if (svg.length > 1000) break;
        await delay(500);
      }
      if (!svg) throw new Error("No rendered SVG diagram was found in the viewer");
      await writeFile(output, svg, "utf8");
    } finally {
      cdp.close();
    }
  } finally {
    if (process.platform === "win32") {
      spawn("taskkill", ["/PID", String(browser.pid), "/T", "/F"], {
        stdio: "ignore",
      });
    } else {
      browser.kill();
    }
    await delay(1000);
    await rm(profile, { recursive: true, force: true }).catch(() => {});
  }
}

await main();
