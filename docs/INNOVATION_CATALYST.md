# Innovation Catalyst CLI Guide

The Innovation Catalyst framework brings structured brainstorming to AURA CLI. Use 8 proven techniques to generate creative solutions to any problem.

## Quick Start

```bash
# Start a simple innovation session
python3 main.py innovate start "How to improve user onboarding?"

# Start with immediate phase execution
python3 main.py innovate start "How to reduce API latency?" --execute-phase divergence

# List all sessions
python3 main.py innovate list

# Show session details
python3 main.py innovate show <session_id>

# Export results
python3 main.py innovate export <session_id> --format markdown
```

## Brainstorming Techniques

The Innovation Catalyst includes 8 proven techniques:

| Technique | Description |
|-----------|-------------|
| **SCAMPER** | Substitute, Combine, Adapt, Modify, Put to other uses, Eliminate, Reverse |
| **Six Thinking Hats** | Parallel thinking from different perspectives (facts, emotions, caution, benefits, creativity, process) |
| **Mind Mapping** | Visual brainstorming with hierarchical idea mapping |
| **Reverse Brainstorming** | Identify problems to find solutions |
| **Worst Idea** | Invert bad ideas to find good ones |
| **Lotus Blossom** | Expand ideas in a grid pattern |
| **Starbursting** | Generate questions from different angles (who, what, when, where, why, how) |
| **BIA** | Bottleneck Identification & Analysis - Find constraints |

View all techniques:
```bash
python3 main.py innovate techniques
```

## The 5-Phase Innovation Process

1. **Immersion** - Deep understanding of the problem
2. **Divergence** - Generate many ideas using multiple techniques
3. **Convergence** - Evaluate and select the best ideas
4. **Incubation** - Let ideas develop (simulated reflection)
5. **Transformation** - Convert ideas into actionable tasks

## Command Reference

### `innovate start`

Start a new innovation session.

```bash
# Basic usage
python3 main.py innovate start "Your problem statement"

# With specific techniques
python3 main.py innovate start "Problem" --techniques scamper,six_hats

# Execute a phase immediately
python3 main.py innovate start "Problem" --execute-phase divergence

# With constraints
python3 main.py innovate start "Problem" --constraints '{"max_ideas": 10}'

# JSON output
python3 main.py innovate start "Problem" --output json
```

**Options:**
- `--techniques` - Comma-separated list of techniques (default: scamper,six_hats,mind_map)
- `--execute-phase` - Execute a phase immediately (immersion, divergence, convergence, incubation, transformation)
- `--constraints` - JSON string with constraints
- `--batch` - Path to file with multiple problems (one per line)
- `--output` - Output format: table or json

### `innovate list`

List all innovation sessions.

```bash
python3 main.py innovate list
python3 main.py innovate list --json
python3 main.py innovate list --limit 10
```

### `innovate show`

Show details of a specific session.

```bash
python3 main.py innovate show <session_id>
python3 main.py innovate show <session_id> --show-ideas
python3 main.py innovate show <session_id> --json
```

### `innovate resume`

Resume a session at a specific phase.

```bash
python3 main.py innovate resume <session_id>
python3 main.py innovate resume <session_id> --phase convergence
```

### `innovate export`

Export session results to markdown or JSON.

```bash
python3 main.py innovate export <session_id> --format markdown
python3 main.py innovate export <session_id> --format json
python3 main.py innovate export <session_id> --output report.md
```

### `innovate techniques`

List all available brainstorming techniques.

```bash
python3 main.py innovate techniques
python3 main.py innovate techniques --json
```

## Batch Mode

Process multiple problems at once:

```bash
# Create a file with problems
cat > problems.txt << 'EOF'
# Product improvements
How to improve signup conversion?
How to reduce churn?
How to improve documentation?
EOF

# Run batch innovation
python3 main.py innovate start --batch problems.txt --execute-phase divergence
```

Output shows aggregated statistics across all sessions.

## Constraints

Customize the innovation process with constraints:

```bash
python3 main.py innovate start "Problem" --constraints '{
  "max_ideas": 20,
  "selection_ratio": 0.2,
  "min_novelty": 0.5,
  "min_feasibility": 0.4,
  "diversity_threshold": 0.7
}'
```

## Persistence

Sessions are automatically persisted to the Brain database (`memory/brain.db`):

- Sessions survive CLI restarts
- Available across different terminal sessions
- Stored with full state and output

## Examples

### Example 1: Feature Ideation

```bash
# Start session and generate ideas
python3 main.py innovate start "What features should we add to the dashboard?" \
  --techniques scamper,mind_map,star \
  --execute-phase divergence

# List to get session ID
python3 main.py innovate list

# Export results for sharing
python3 main.py innovate export <session_id> --output features.md
```

### Example 2: Problem Solving Workshop

```bash
# Create batch file with team problems
cat > workshop.txt << 'EOF'
How to improve code review turnaround?
How to reduce production incidents?
How to improve cross-team communication?
How to make deployments safer?
EOF

# Run all problems
python3 main.py innovate start --batch workshop.txt --execute-phase divergence

# Show summary
python3 main.py innovate list
```

### Example 3: Architecture Brainstorming

```bash
# Use BIA and Reverse techniques for architecture
python3 main.py innovate start "How should we redesign the data pipeline?" \
  --techniques bia,reverse,scamper \
  --execute-phase divergence

# Resume for convergence
python3 main.py innovate resume <session_id> --phase convergence

# Export final results
python3 main.py innovate export <session_id> --format markdown
```

## Integration with Workflow

### From Goals

```bash
# Add innovation as a goal
python3 main.py goal add "innovate: How to improve X?"

# Run goals queue
python3 main.py goal run
```

### Export to Tasks

After generating ideas with `innovate`, export the results and use them to create implementation goals:

```bash
# Export ideas
python3 main.py innovate export <session_id> --format json

# Create goals from selected ideas
python3 main.py goal add "Implement idea A from innovation session"
```

## Tips

1. **Start broad**: Use divergence phase first to generate many ideas
2. **Mix techniques**: Different techniques produce different types of ideas
3. **Iterate**: Resume sessions at different phases to refine results
4. **Batch related problems**: Process multiple related problems together
5. **Export regularly**: Save results before moving to implementation

## Troubleshooting

### Session not found

Sessions are persisted but require the Brain to be initialized. If a session is not found:
- Verify the session ID is correct
- Check that the Brain database exists (`memory/brain.db`)

### Invalid techniques

Use `python3 main.py innovate techniques` to see valid technique names.

### Constraints JSON errors

Ensure constraints are valid JSON:
```bash
# Valid
'{"max_ideas": 10}'

# Invalid (missing quotes)
'{max_ideas: 10}'
```
