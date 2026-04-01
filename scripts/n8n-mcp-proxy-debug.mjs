#!/usr/bin/env node
import { appendFileSync } from 'fs';
const LOG = '/tmp/n8n-proxy-debug.log';
const log = (msg) => appendFileSync(LOG, `${new Date().toISOString()} ${msg}\n`);

log('=== PROXY STARTED ===');
log(`env N8N_MCP_URL=${process.env.N8N_MCP_URL}`);
log(`env N8N_MCP_TOKEN=${process.env.N8N_MCP_TOKEN ? 'SET' : 'UNSET'}`);
log(`node ${process.version}`);
log(`argv: ${process.argv.join(' ')}`);

const N8N_URL = process.env.N8N_MCP_URL || 'http://localhost:5678/mcp-server/http';
const N8N_TOKEN = process.env.N8N_MCP_TOKEN || '';

const baseHeaders = {
  'Content-Type': 'application/json',
  'Accept': 'application/json, text/event-stream',
};
if (N8N_TOKEN) baseHeaders['Authorization'] = `Bearer ${N8N_TOKEN}`;

let sessionId = null;

process.on('uncaughtException', (err) => { log(`UNCAUGHT: ${err.stack}`); });
process.on('unhandledRejection', (err) => { log(`REJECTION: ${err?.stack || err}`); });

function sanitizeTools(tools) {
  return tools.map((t) => ({
    name: t.name,
    description: t.description,
    inputSchema: t.inputSchema,
  }));
}

async function sendToN8n(message) {
  const reqHeaders = { ...baseHeaders };
  if (sessionId) reqHeaders['mcp-session-id'] = sessionId;
  log(`SEND to n8n: ${message.method || 'notification'} id=${message.id}`);

  const res = await fetch(N8N_URL, {
    method: 'POST',
    headers: reqHeaders,
    body: JSON.stringify(message),
  });

  log(`RECV from n8n: status=${res.status} ct=${res.headers.get('content-type')}`);
  const sid = res.headers.get('mcp-session-id');
  if (sid) sessionId = sid;

  const ct = res.headers.get('content-type') || '';
  if (ct.includes('text/event-stream')) {
    const text = await res.text();
    const results = [];
    for (const line of text.split('\n')) {
      if (line.startsWith('data: ')) {
        try { results.push(JSON.parse(line.slice(6))); } catch {}
      }
    }
    log(`PARSED ${results.length} SSE messages`);
    return results;
  }
  const json = await res.json();
  return [json];
}

async function handleMessage(message) {
  log(`HANDLE: method=${message.method} id=${message.id}`);

  if (!message.id && message.method) {
    if (message.method === 'notifications/initialized') {
      sendToN8n({
        jsonrpc: '2.0', id: 'bg-init', method: 'initialize',
        params: { protocolVersion: '2024-11-05', capabilities: {},
          clientInfo: { name: 'n8n-mcp-proxy', version: '1.0.0' } }
      }).then(() => {
        log('BG init done, sending notifications/initialized');
        return sendToN8n({ jsonrpc: '2.0', method: 'notifications/initialized' });
      }).catch((e) => log(`BG init error: ${e.message}`));
    }
    return;
  }

  if (message.method === 'initialize') {
    const resp = {
      jsonrpc: '2.0', id: message.id,
      result: {
        protocolVersion: '2024-11-05',
        capabilities: { tools: { listChanged: false } },
        serverInfo: { name: 'n8n-mcp-proxy', version: '1.0.0' },
      },
    };
    const out = JSON.stringify(resp) + '\n';
    log(`WRITE initialize response: ${out.length} bytes`);
    process.stdout.write(out);
    return;
  }

  try {
    const responses = await sendToN8n(message);
    for (const resp of responses) {
      if (message.method === 'tools/list' && resp?.result?.tools) {
        resp.result.tools = sanitizeTools(resp.result.tools);
        log(`SANITIZED ${resp.result.tools.length} tools`);
      }
      const out = JSON.stringify(resp) + '\n';
      log(`WRITE response: ${out.length} bytes`);
      process.stdout.write(out);
    }
  } catch (err) {
    log(`ERROR: ${err.message}`);
    process.stdout.write(JSON.stringify({
      jsonrpc: '2.0', id: message.id,
      error: { code: -32000, message: err.message },
    }) + '\n');
  }
}

let buffer = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => {
  log(`STDIN chunk: ${chunk.length} bytes`);
  buffer += chunk;
  let idx;
  while ((idx = buffer.indexOf('\n')) !== -1) {
    const line = buffer.slice(0, idx).trim();
    buffer = buffer.slice(idx + 1);
    if (!line) continue;
    log(`STDIN line: ${line.substring(0, 100)}...`);
    try { handleMessage(JSON.parse(line)); } catch (e) { log(`PARSE ERROR: ${e.message}`); }
  }
});
process.stdin.on('end', () => {
  log('STDIN ended');
  setTimeout(() => process.exit(0), 2000);
});
process.stdin.resume();
log('PROXY READY, waiting for input');
