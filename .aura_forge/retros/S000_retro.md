# Retro — Sprint S000

Date: 2026-03-18

---

## Wins

- **Complete Forge scaffold created** — All 27 files in place across schemas, templates, backlog, examples, and indexes
- **Zero runtime contract disruption** — No changes to CLI, queue behavior, or safety policy
- **Clear artifact lifecycle established** — Promotion rules defined from inbox → done
- **Design pass templates ready** — Six structured thinking methods available
- **Worked examples created** — Three real stories (AF-STORY-0001/0002/0003) demonstrate the model

---

## Friction

- **Some overlap between plans/ and Forge story artifacts** — Need stricter rule for when retros produce capability deltas vs. when stories link to existing plans
- **Template verbosity** — Some templates may be too heavy for quick ideas; need lighter-weight option
- **No automation yet** — All promotion is manual; this is intentional for phase 1 but creates overhead

---

## Lessons

- **Forge works best as upstream cognition, not orchestration** — Keeping it separate from runtime code was the right call
- **Story artifacts should link into plans/, not duplicate plans/** — The relationship needs clearer definition
- **Contract impact and safety impact declarations are valuable** — Even as documentation, they force explicit thinking
- **File-first approach reduces risk** — No CLI changes means no snapshot test breakage, no help text updates needed

---

## Capability Deltas Proposed

- **AF-DELTA-0001**: Add capability-gap index generation
- **AF-DELTA-0002**: Add queue-candidate review template  
- **AF-DELTA-0003**: Add lightweight story template for quick ideas

---

## Heuristics To Promote

| Pattern | Rationale | Status |
|---------|-----------|--------|
| Every story touching CLI must declare contract impact | CLI behavior is contract-bound with generated docs and snapshot checks | Draft |
| Every story touching autonomous apply must declare safety impact | Overwrite safety is centralized in core/file_tools.py and policy-bound | Draft |
| Forge stories link to plans/, they don't replace plans/ | Implementation planning stays in plans/; Forge handles upstream shaping | Draft |
| Start file-first; add CLI only after artifact model stabilizes | CLI changes impose maintenance cost via generated docs and snapshots | Draft |

---

## Notes

S000 achieved its goal of establishing the Forge scaffold without runtime changes. The system is ready for actual use. Next sprint should focus on making Forge output actionable (queue-ready intelligence) while keeping the manual-promotion safety boundary.
