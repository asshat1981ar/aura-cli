// AURA Knowledge Graph — Neo4j Schema
// Run once to initialize the graph database schema.
//
// Usage:
//   cat infra/neo4j/schema.cypher | cypher-shell -u neo4j -p <password>
//   OR mount as /docker-entrypoint-initdb.d/schema.cypher in docker-compose

// ═══════════════════════════════════════════════════════════════════════════
// Uniqueness constraints
// ═══════════════════════════════════════════════════════════════════════════

CREATE CONSTRAINT IF NOT EXISTS FOR (f:File) REQUIRE f.path IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (fn:Function) REQUIRE fn.qualified_name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (c:Class) REQUIRE c.qualified_name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (m:Module) REQUIRE m.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (g:Goal) REQUIRE g.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (a:Agent) REQUIRE a.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (cy:Cycle) REQUIRE cy.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (mem:Memory) REQUIRE mem.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (env:Environment) REQUIRE env.name IS UNIQUE;

// ═══════════════════════════════════════════════════════════════════════════
// Node type documentation (labels and expected properties)
// ═══════════════════════════════════════════════════════════════════════════
//
// (:File {path, language, size_bytes, last_modified, line_count})
// (:Function {qualified_name, name, file_path, line_start, line_end, complexity, is_async})
// (:Class {qualified_name, name, file_path, line_start, line_end, method_count})
// (:Module {name, path, package})
// (:Goal {id, text, status, priority, created_at, completed_at, score})
// (:Agent {name, type, environment, capabilities, status, registered_at})
// (:Skill {name, type, weight, enabled})
// (:Cycle {id, goal_id, outcome, score, phase, timestamp, elapsed_ms})
// (:Memory {id, content_hash, tier, timestamp, relevance_score})
// (:Environment {name, cli_type, workspace_root, status})
//
// ═══════════════════════════════════════════════════════════════════════════
// Relationship type documentation
// ═══════════════════════════════════════════════════════════════════════════
//
// (:File)-[:IMPORTS {line}]->(:File)
// (:Function)-[:DEFINED_IN]->(:File)
// (:Class)-[:DEFINED_IN]->(:File)
// (:Function)-[:CALLS {count}]->(:Function)
// (:Function)-[:MEMBER_OF]->(:Class)
// (:Class)-[:INHERITS]->(:Class)
// (:Module)-[:CONTAINS]->(:File)
// (:Goal)-[:EXECUTED_BY]->(:Agent)
// (:Goal)-[:USED_SKILL {weight}]->(:Skill)
// (:Goal)-[:DEPENDS_ON]->(:Goal)
// (:Cycle)-[:FOR_GOAL]->(:Goal)
// (:Cycle)-[:PRODUCED]->(:Memory)
// (:Agent)-[:BELONGS_TO]->(:Environment)
// (:Agent)-[:HAS_CAPABILITY {proficiency}]->(:Skill)
//

// ═══════════════════════════════════════════════════════════════════════════
// Performance indexes
// ═══════════════════════════════════════════════════════════════════════════

CREATE INDEX IF NOT EXISTS FOR (f:File) ON (f.language);
CREATE INDEX IF NOT EXISTS FOR (g:Goal) ON (g.status);
CREATE INDEX IF NOT EXISTS FOR (g:Goal) ON (g.created_at);
CREATE INDEX IF NOT EXISTS FOR (cy:Cycle) ON (cy.timestamp);
CREATE INDEX IF NOT EXISTS FOR (cy:Cycle) ON (cy.goal_id);
CREATE INDEX IF NOT EXISTS FOR (a:Agent) ON (a.type);
CREATE INDEX IF NOT EXISTS FOR (a:Agent) ON (a.environment);
CREATE INDEX IF NOT EXISTS FOR (s:Skill) ON (s.type);
CREATE INDEX IF NOT EXISTS FOR (mem:Memory) ON (mem.tier);
CREATE INDEX IF NOT EXISTS FOR (mem:Memory) ON (mem.timestamp);
