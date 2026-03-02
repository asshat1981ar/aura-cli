import process from "node:process";
import { spawnSync } from "node:child_process";

const SCHEMA_VERSION = 1;

async function readStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString("utf8");
}

function normalizeDecision(raw) {
  const decision = raw && typeof raw === "object" ? raw : {};
  const status = ["allow", "block", "revise"].includes(decision.status)
    ? decision.status
    : "revise";

  return {
    schema_version: SCHEMA_VERSION,
    decision_id:
      typeof decision.decision_id === "string" && decision.decision_id.trim()
        ? decision.decision_id
        : `beads-${Date.now()}`,
    status,
    summary:
      typeof decision.summary === "string" && decision.summary.trim()
        ? decision.summary
        : "BEADS decision generated without summary.",
    rationale: Array.isArray(decision.rationale)
      ? decision.rationale.filter((item) => typeof item === "string")
      : [],
    required_constraints: Array.isArray(decision.required_constraints)
      ? decision.required_constraints.filter((item) => typeof item === "string")
      : [],
    required_skills: Array.isArray(decision.required_skills)
      ? decision.required_skills.filter((item) => typeof item === "string")
      : [],
    required_tests: Array.isArray(decision.required_tests)
      ? decision.required_tests.filter((item) => typeof item === "string")
      : [],
    follow_up_goals: Array.isArray(decision.follow_up_goals)
      ? decision.follow_up_goals.filter((item) => typeof item === "string")
      : [],
    stop_reason:
      typeof decision.stop_reason === "string" ? decision.stop_reason : null,
  };
}

function resultOk(decision, startedAt, stderr = null) {
  return {
    schema_version: SCHEMA_VERSION,
    ok: true,
    status: "ok",
    decision,
    error: null,
    stderr,
    duration_ms: Date.now() - startedAt,
  };
}

function resultError(error, startedAt, stderr = null) {
  return {
    schema_version: SCHEMA_VERSION,
    ok: false,
    status: "error",
    decision: null,
    error,
    stderr,
    duration_ms: Date.now() - startedAt,
  };
}

async function invokeBeads(payload) {
  const version = spawnSync("bd", ["--json", "version"], { encoding: "utf8" });
  if (version.status !== 0) {
    const detail = (version.stderr || version.stdout || "").trim();
    throw new Error(detail ? `beads_cli_unavailable: ${detail}` : "beads_cli_unavailable");
  }

  const info = spawnSync("bd", ["info", "--json"], { encoding: "utf8" });
  const detail = (info.stderr || info.stdout || "").trim();
  if (info.status !== 0) {
    throw new Error(detail ? `beads_database_missing: ${detail}` : "beads_database_missing");
  }

  throw new Error("beads_decision_adapter_missing");
}

async function main() {
  const startedAt = Date.now();

  try {
    const input = await readStdin();
    const payload = JSON.parse(input);
    const decision = await invokeBeads(payload);
    process.stdout.write(`${JSON.stringify(resultOk(decision, startedAt))}\n`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    process.stdout.write(`${JSON.stringify(resultError(message, startedAt, message))}\n`);
  }
}

main();
