import fs from "node:fs";
import path from "node:path";
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

function resolveBdCommand() {
  if (process.env.BEADS_CLI && process.env.BEADS_CLI.trim()) {
    return process.env.BEADS_CLI.trim();
  }
  if (process.env.BD_COMMAND && process.env.BD_COMMAND.trim()) {
    return process.env.BD_COMMAND.trim();
  }
  const localCommand = localBdCommand();
  if (fs.existsSync(localCommand)) {
    return localCommand;
  }

  return "bd";
}

function localBdCommand() {
  return process.platform === "win32"
    ? path.join(process.cwd(), "node_modules", ".bin", "bd.cmd")
    : path.join(process.cwd(), "node_modules", ".bin", "bd");
}

function isLocalBdCommand(command) {
  if (!command || command === "bd") {
    return false;
  }
  return path.resolve(command) === path.resolve(localBdCommand());
}

function safeParseJson(text) {
  if (typeof text !== "string" || !text.trim()) {
    return null;
  }
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function isCliUnavailable(problem) {
  return typeof problem === "string" && (
    problem.includes("ENOENT") ||
    problem.includes("not found")
  );
}

function runBdJson(args, timeoutMs = 1500) {
  const command = resolveBdCommand();
  const commandArgs = isLocalBdCommand(command) ? ["--no-daemon", ...args] : Array.from(args);
  let proc = spawnSync(command, commandArgs, {
    encoding: "utf8",
    cwd: process.cwd(),
    env: process.env,
    timeout: timeoutMs,
  });
  const fallbackCommand = localBdCommand();
  if (
    proc.error instanceof Error &&
    proc.error.code === "ENOENT" &&
    command === "bd" &&
    fs.existsSync(fallbackCommand)
  ) {
    const fallbackArgs = ["--no-daemon", ...args];
    proc = spawnSync(fallbackCommand, fallbackArgs, {
      encoding: "utf8",
      cwd: process.cwd(),
      env: process.env,
      timeout: timeoutMs,
    });
  }
  const stdout = typeof proc.stdout === "string" ? proc.stdout.trim() : "";
  const stderr = typeof proc.stderr === "string" ? proc.stderr.trim() : "";
  const error = proc.error instanceof Error ? proc.error.message : null;
  const status = typeof proc.status === "number" ? proc.status : (error ? 1 : 0);
  const json = safeParseJson(stdout);

  return {
    command,
    args: commandArgs,
    ok: status === 0 && json !== null,
    status,
    stdout,
    stderr,
    error,
    json,
  };
}

function listFromPayload(payload, preferredKey) {
  if (Array.isArray(payload)) {
    return payload;
  }
  if (!payload || typeof payload !== "object") {
    return [];
  }
  if (preferredKey && Array.isArray(payload[preferredKey])) {
    return payload[preferredKey];
  }
  for (const key of ["ready", "items", "issues", "results"]) {
    if (Array.isArray(payload[key])) {
      return payload[key];
    }
  }
  return [];
}

function formatIssueGoal(item) {
  if (!item || typeof item !== "object") {
    return null;
  }
  const id = typeof item.id === "string" && item.id.trim() ? item.id.trim() : null;
  const title = typeof item.title === "string" && item.title.trim()
    ? item.title.trim()
    : (typeof item.summary === "string" && item.summary.trim() ? item.summary.trim() : null);
  if (id && title) {
    return `bead:${id}: ${title}`;
  }
  if (id) {
    return `bead:${id}`;
  }
  return title;
}

function uniqueStrings(values) {
  const seen = new Set();
  const output = [];
  for (const value of values) {
    if (typeof value !== "string" || !value.trim()) {
      continue;
    }
    const trimmed = value.trim();
    if (seen.has(trimmed)) {
      continue;
    }
    seen.add(trimmed);
    output.push(trimmed);
  }
  return output;
}

function stringList(value) {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter((item) => typeof item === "string" && item.trim())
    .map((item) => item.trim());
}

function goalTypeRequiredSkills(goalType) {
  switch (goalType) {
    case "bug_fix":
      return ["test_and_observe", "lint"];
    case "feature":
      return ["lint", "architecture_validator"];
    case "refactor":
      return ["lint", "test_and_observe"];
    case "security":
      return ["security_scanner", "lint"];
    case "docs":
      return ["doc_generator"];
    default:
      return ["lint"];
  }
}

function goalTypeRequiredTests(goalType) {
  switch (goalType) {
    case "bug_fix":
      return [
        "Run targeted regression tests for the failing path.",
        "Re-run the narrowest pytest scope that covers the fix.",
      ];
    case "feature":
      return ["Run focused tests covering the new or changed behavior."];
    case "refactor":
      return [
        "Run targeted tests for the refactored modules.",
        "Verify no contract regressions in the touched path.",
      ];
    case "security":
      return ["Run focused verification for auth, permissions, and input handling touched by this change."];
    case "docs":
      return [];
    default:
      return ["Run the most relevant targeted verification for the touched code path."];
  }
}

function capabilitySignals(payload) {
  const activeContext = payload && typeof payload.active_context === "object" && payload.active_context
    ? payload.active_context
    : {};
  const capabilityPlan = activeContext && typeof activeContext.capability_plan === "object" && activeContext.capability_plan
    ? activeContext.capability_plan
    : {};
  const capabilityGoalQueue = activeContext && typeof activeContext.capability_goal_queue === "object" && activeContext.capability_goal_queue
    ? activeContext.capability_goal_queue
    : {};
  const capabilityProvisioning = activeContext && typeof activeContext.capability_provisioning === "object" && activeContext.capability_provisioning
    ? activeContext.capability_provisioning
    : {};

  const missingSkills = listFromPayload({ items: capabilityPlan.missing_skills }, "items")
    .filter((item) => typeof item === "string");
  const recommendedSkills = listFromPayload({ items: capabilityPlan.recommended_skills }, "items")
    .filter((item) => typeof item === "string");
  const queuedGoals = listFromPayload({ items: capabilityGoalQueue.queued }, "items")
    .filter((item) => typeof item === "string");
  const provisioningActions = listFromPayload({ items: capabilityPlan.provisioning_actions }, "items");
  const provisioningStatus = listFromPayload({ items: capabilityProvisioning.actions }, "items");

  return {
    missingSkills,
    recommendedSkills,
    queuedGoals,
    provisioningActions,
    provisioningStatus,
  };
}

function developmentSignals(payload) {
  const developmentContext = payload && typeof payload.development_context === "object" && payload.development_context
    ? payload.development_context
    : {};
  const prototypeInventory = developmentContext && typeof developmentContext.prototype_inventory === "object" && developmentContext.prototype_inventory
    ? developmentContext.prototype_inventory
    : {};
  const weaknesses = Array.isArray(developmentContext.weaknesses)
    ? developmentContext.weaknesses.filter((item) => item && typeof item === "object")
    : [];

  return {
    targetSubsystem:
      typeof developmentContext.target_subsystem === "string" && developmentContext.target_subsystem.trim()
        ? developmentContext.target_subsystem.trim()
        : "general",
    canonicalPath:
      typeof developmentContext.canonical_path === "string" && developmentContext.canonical_path.trim()
        ? developmentContext.canonical_path.trim()
        : null,
    overlapClassification:
      typeof developmentContext.overlap_classification === "string" && developmentContext.overlap_classification.trim()
        ? developmentContext.overlap_classification.trim()
        : "unknown",
    validationStatus:
      typeof developmentContext.validation_status === "string" && developmentContext.validation_status.trim()
        ? developmentContext.validation_status.trim()
        : "unknown",
    deprecatedPaths: stringList(prototypeInventory.deprecated_paths),
    canonicalPaths: stringList(prototypeInventory.canonical_paths),
    validationNotes: stringList(prototypeInventory.validation_notes),
    weaknesses,
  };
}

function goalMentionsPath(goalText, filePath) {
  if (typeof goalText !== "string" || !goalText.trim() || typeof filePath !== "string" || !filePath.trim()) {
    return false;
  }
  const goalLower = goalText.toLowerCase();
  const normalizedPath = filePath.replace(/\\/g, "/").toLowerCase();
  const baseName = normalizedPath.split("/").pop();
  return goalLower.includes(normalizedPath) || (baseName ? goalLower.includes(baseName) : false);
}

function isRetirementGoal(goalText) {
  const goalLower = typeof goalText === "string" ? goalText.toLowerCase() : "";
  return [
    "retire",
    "remove",
    "delete",
    "deprecate",
    "replace",
    "migrate",
    "port",
  ].some((token) => goalLower.includes(token));
}

function buildDecision(payload, beadsSignals) {
  const goal = typeof payload.goal === "string" ? payload.goal.trim() : "";
  const goalType = typeof payload.goal_type === "string" ? payload.goal_type : "default";
  const queueSummary = payload && typeof payload.queue_summary === "object" && payload.queue_summary
    ? payload.queue_summary
    : {};
  const planningGaps = [];
  const rationale = [];
  const requiredConstraints = [
    "Keep changes scoped to the active goal.",
    "LoopOrchestrator remains the sole execution authority.",
  ];
  const followUpGoals = [];
  const requiredRemediation = [];
  const capability = capabilitySignals(payload);
  const development = developmentSignals(payload);
  const requiredSkills = [...goalTypeRequiredSkills(goalType), ...capability.recommendedSkills, ...capability.missingSkills];
  const requiredTests = [...goalTypeRequiredTests(goalType)];
  let targetSubsystem = development.targetSubsystem;
  let canonicalPath = development.canonicalPath;
  let overlapClassification = development.overlapClassification;
  let validationStatus = development.validationStatus;

  if (!goal) {
    return normalizeDecision({
      status: "block",
      summary: "Blocked because the BEADS bridge did not receive a goal.",
      rationale: ["The canonical BEADS payload requires a non-empty goal string."],
      required_constraints: requiredConstraints,
      required_skills: uniqueStrings(requiredSkills),
      required_tests: uniqueStrings(requiredTests),
      follow_up_goals: [],
      target_subsystem: targetSubsystem,
      canonical_path: canonicalPath,
      overlap_classification: overlapClassification,
      validation_status: validationStatus,
      required_remediation: [],
      stop_reason: "invalid_goal",
    });
  }

  if (!payload.prd_context) {
    planningGaps.push("PRD context");
  }
  if (!payload.conductor_track) {
    planningGaps.push("conductor track context");
  }

  if (payload.prd_context && typeof payload.prd_context === "object") {
    rationale.push(`PRD context loaded from ${payload.prd_context.path || "configured source"}.`);
  }
  if (payload.conductor_track && typeof payload.conductor_track === "object") {
    rationale.push(`Conductor track context loaded from ${payload.conductor_track.path || payload.conductor_track.track_id || "configured source"}.`);
  }

  if (typeof queueSummary.pending_count === "number") {
    rationale.push(`Queue currently has ${queueSummary.pending_count} pending goal(s).`);
  }

  if (beadsSignals.info.ok && beadsSignals.info.json && typeof beadsSignals.info.json === "object") {
    const issueCount = Number(beadsSignals.info.json.issue_count || 0);
    const mode = typeof beadsSignals.info.json.mode === "string" ? beadsSignals.info.json.mode : "unknown";
    rationale.push(`BEADS runtime reported ${issueCount} tracked issue(s) in ${mode} mode.`);
  } else if (beadsSignals.ready.ok) {
    rationale.push("BEADS ready-work lookup completed even though runtime info was unavailable.");
  } else if (beadsSignals.info.problem) {
    rationale.push("BEADS database signals were unavailable; the bridge fell back to payload-only decision synthesis.");
    requiredConstraints.push("Treat BEADS ready-work signals as unavailable until BEADS database health is restored.");
  }

  if (beadsSignals.ready.problem) {
    rationale.push("Ready-work lookup did not complete, so no BEADS follow-up work was imported.");
  }
  for (const item of beadsSignals.ready.goals) {
    followUpGoals.push(item);
  }

  if (capability.missingSkills.length > 0) {
    rationale.push(`Capability analysis still reports missing skills: ${capability.missingSkills.join(", ")}.`);
  }
  if (capability.provisioningActions.length > 0 || capability.provisioningStatus.length > 0) {
    rationale.push("Capability provisioning work is still pending for this goal.");
  }
  for (const goalText of capability.queuedGoals) {
    followUpGoals.push(goalText);
  }

  let status = "allow";
  let summary = "Proceed with a scoped BEADS-backed execution plan.";
  let stopReason = null;
  const setRevise = (nextSummary, nextStopReason) => {
    if (status === "block") {
      return;
    }
    status = "revise";
    summary = nextSummary;
    if (!stopReason) {
      stopReason = nextStopReason;
    }
  };
  const setBlock = (nextSummary, nextStopReason) => {
    status = "block";
    summary = nextSummary;
    stopReason = nextStopReason;
  };

  if (
    capability.missingSkills.length > 0 ||
    capability.provisioningActions.length > 0 ||
    capability.queuedGoals.length > 0
  ) {
    setRevise("Revise before execution: capability prerequisites are still pending.", "capability_prerequisites_missing");
    requiredConstraints.push("Resolve or queue missing capability work before broad implementation changes.");
    requiredRemediation.push("Resolve the queued capability prerequisites before broad implementation changes.");
  } else if (planningGaps.length > 0) {
    setRevise("Revise before execution: canonical planning context is incomplete.", "planning_context_incomplete");
    rationale.push(`Missing planning inputs: ${planningGaps.join(", ")}.`);
    requiredConstraints.push("Refresh the canonical PRD and conductor context before large-scope execution.");
    requiredRemediation.push("Refresh the canonical PRD and conductor context before large-scope execution.");
  }

  for (const note of development.validationNotes) {
    rationale.push(note);
  }
  for (const weakness of development.weaknesses) {
    if (typeof weakness.summary === "string" && weakness.summary.trim()) {
      rationale.push(`Weakness scan: ${weakness.summary.trim()}`);
    }
  }

  if (targetSubsystem === "recursive_self_improvement") {
    if (canonicalPath) {
      rationale.push(`Canonical RSI runtime path: ${canonicalPath}.`);
    }
    if (development.deprecatedPaths.length > 0) {
      rationale.push(`Deprecated RSI prototype paths remain present: ${development.deprecatedPaths.join(", ")}.`);
    }
    requiredConstraints.push("Do not add or restore runtime logic in deprecated RSI prototype paths.");
    requiredTests.push("Run tests/test_recursive_improvement.py and tests/test_fitness.py.");
    requiredTests.push("Run tests/test_evolution_loop_rsi.py and tests/test_rsi_integration_verification.py.");

    const touchesDeprecatedPath = development.deprecatedPaths.some((filePath) => goalMentionsPath(goal, filePath));
    if (touchesDeprecatedPath && !isRetirementGoal(goal)) {
      setBlock(
        "Blocked: the goal targets a retired recursive self-improvement prototype path.",
        "deprecated_recursive_self_improvement_path",
      );
      requiredRemediation.push(`Port the work to ${canonicalPath || "the canonical RSI runtime path"} instead of reviving a retired prototype entrypoint.`);
    } else if (touchesDeprecatedPath) {
      rationale.push("Goal references a deprecated RSI path, but the wording suggests a retirement or migration task.");
    }

    if (overlapClassification !== "canonical_only") {
      setRevise(
        "Revise before execution: recursive self-improvement work must converge on the canonical path.",
        "recursive_self_improvement_overlap",
      );
      requiredRemediation.push("Retire or shrink overlapping RSI prototype entrypoints before expanding the workflow.");
    }

    if (validationStatus !== "validated" && validationStatus !== "not_applicable") {
      setRevise(
        "Revise before execution: recursive self-improvement validation is still incomplete.",
        "recursive_self_improvement_validation_pending",
      );
      requiredRemediation.push("Complete the non-dry-run 50-cycle RSI audit in an isolated workspace before broadening autonomous mutation scope.");
    }
  }

  return normalizeDecision({
    status,
    summary,
    rationale,
    required_constraints: uniqueStrings(requiredConstraints),
    required_skills: uniqueStrings(requiredSkills),
    required_tests: uniqueStrings(requiredTests),
    follow_up_goals: uniqueStrings(followUpGoals),
    target_subsystem: targetSubsystem,
    canonical_path: canonicalPath,
    overlap_classification: overlapClassification,
    validation_status: validationStatus,
    required_remediation: uniqueStrings(requiredRemediation),
    stop_reason: stopReason,
  });
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
    target_subsystem:
      typeof decision.target_subsystem === "string" && decision.target_subsystem.trim()
        ? decision.target_subsystem.trim()
        : "general",
    canonical_path:
      typeof decision.canonical_path === "string" && decision.canonical_path.trim()
        ? decision.canonical_path.trim()
        : null,
    overlap_classification:
      typeof decision.overlap_classification === "string" && decision.overlap_classification.trim()
        ? decision.overlap_classification.trim()
        : "unknown",
    validation_status:
      typeof decision.validation_status === "string" && decision.validation_status.trim()
        ? decision.validation_status.trim()
        : "unknown",
    required_remediation: Array.isArray(decision.required_remediation)
      ? decision.required_remediation.filter((item) => typeof item === "string")
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
  const info = runBdJson(["info", "--json"], 1200);
  const infoProblem = info.ok
    ? null
    : (info.error || info.stderr || info.stdout || "beads_database_unavailable");

  let readyGoals = [];
  let readyProblem = null;
  let readyOk = false;
  const infoJson = info.ok && info.json && typeof info.json === "object" ? info.json : null;
  const issueCount = infoJson && Number.isFinite(Number(infoJson.issue_count))
    ? Number(infoJson.issue_count)
    : 0;

  if (!info.ok || issueCount > 0) {
    const ready = runBdJson(["ready", "--json"], 12000);
    if (ready.ok) {
      readyOk = true;
      readyGoals = listFromPayload(ready.json, "ready")
        .map((item) => formatIssueGoal(item))
        .filter((item) => typeof item === "string");
    } else {
      readyProblem = ready.error || ready.stderr || ready.stdout || "ready_lookup_failed";
    }
  }

  if (!info.ok && !readyOk && (isCliUnavailable(infoProblem) || isCliUnavailable(readyProblem))) {
    const detail = readyProblem || infoProblem || "";
    throw new Error(detail ? `beads_cli_unavailable: ${detail}` : "beads_cli_unavailable");
  }

  return buildDecision(payload, {
    info: {
      ok: info.ok,
      json: infoJson,
      problem: infoProblem,
    },
    ready: {
      ok: readyOk,
      goals: readyGoals,
      problem: readyProblem,
    },
  });
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
    const code = message.includes("beads_cli_unavailable")
      ? "capability_unavailable"
      : message;
    process.stdout.write(`${JSON.stringify(resultError(code, startedAt, message))}\n`);
  }
}

main();
