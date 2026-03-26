# Story: {{title}}

**ID:** {{id}}  
**Lane:** {{lane}}  
**Status:** {{status}}  
**Sprint:** {{sprint_link}}

---

## Goal

### User Goal
{{user_goal}}

### System Goal
{{system_goal}}

---

## Context

### Sensed
{{sensed}}

### Retrieved
{{retrieved}}

---

## Design Passes Applied

<!-- List which passes have been run. Typical batch: all passes in one session before promoting. -->
{{design_passes}}

<!-- design_pass_notes: add a `design_pass_notes` mapping below once passes are complete.
     Each key matches a pass name (reverse_engineering, reverse_decomposition, scamper,
     autotriz, six_hats). See .aura_forge/schemas/story.schema.yaml for the full shape.
     Example:
       design_pass_notes:
         reverse_engineering:
           gap_identified: "X is missing Y"
           observed_rules: [...]
         scamper:
           top_3_design_options: {...}
-->

---

## Constraints
{{constraints}}

---

## Contract Impact

- CLI contract affected: {{cli_contract_affected}}
- Changes: {{contract_changes}}
- Snapshot updates needed: {{snapshot_updates_needed}}

---

## Safety Impact

- Autonomous apply affected: {{autonomous_apply_affected}}
- Overwrite behavior affected: {{overwrite_behavior_affected}}
- Safety notes: {{safety_notes}}

---

## Acceptance Criteria
{{acceptance}}

---

## Risks
{{risks}}

---

## Rollback Plan
{{rollback}}

---

## Expected Outputs
{{outputs}}
