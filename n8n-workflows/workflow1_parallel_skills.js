import { workflow, node, trigger, merge } from '@n8n/workflow-sdk';

const startTrigger = trigger({
  type: 'n8n-nodes-base.manualTrigger',
  version: 1,
  config: { name: 'Start', position: [0, 300] },
  output: [{}]
});

const setProjectRoot = node({
  type: 'n8n-nodes-base.set',
  version: 3.4,
  config: {
    name: 'Set Project Root',
    position: [220, 300],
    parameters: {
      mode: 'manual',
      assignments: {
        assignments: [
          { id: 'pr1', name: 'project_root', value: '/home/westonaaron675/aura-cli', type: 'string' }
        ]
      }
    }
  },
  output: [{ project_root: '/home/westonaaron675/aura-cli' }]
});

const securityScanner = node({
  type: 'n8n-nodes-base.httpRequest',
  version: 4.4,
  config: {
    name: 'Security Scanner',
    position: [440, 60],
    parameters: {
      method: 'POST',
      url: 'http://localhost:8002/call',
      sendHeaders: true,
      specifyHeaders: 'keypair',
      headerParameters: {
        parameters: [
          { name: 'Authorization', value: '=("Bearer " + $env.MCP_API_TOKEN)' }
        ]
      },
      sendBody: true,
      contentType: 'json',
      specifyBody: 'json',
      jsonBody: '={"name":"security_scanner","arguments":{"project_root":"{{ $json.project_root }}"}}'
    },
    continueOnFail: true
  },
  output: [{ skill: 'security_scanner', result: {} }]
});

const testCoverageAnalyzer = node({
  type: 'n8n-nodes-base.httpRequest',
  version: 4.4,
  config: {
    name: 'Test Coverage Analyzer',
    position: [440, 200],
    parameters: {
      method: 'POST',
      url: 'http://localhost:8002/call',
      sendHeaders: true,
      specifyHeaders: 'keypair',
      headerParameters: {
        parameters: [
          { name: 'Authorization', value: '=("Bearer " + $env.MCP_API_TOKEN)' }
        ]
      },
      sendBody: true,
      contentType: 'json',
      specifyBody: 'json',
      jsonBody: '={"name":"test_coverage_analyzer","arguments":{"project_root":"{{ $json.project_root }}"}}'
    },
    continueOnFail: true
  },
  output: [{ skill: 'test_coverage_analyzer', result: {} }]
});

const complexityScorer = node({
  type: 'n8n-nodes-base.httpRequest',
  version: 4.4,
  config: {
    name: 'Complexity Scorer',
    position: [440, 340],
    parameters: {
      method: 'POST',
      url: 'http://localhost:8002/call',
      sendHeaders: true,
      specifyHeaders: 'keypair',
      headerParameters: {
        parameters: [
          { name: 'Authorization', value: '=("Bearer " + $env.MCP_API_TOKEN)' }
        ]
      },
      sendBody: true,
      contentType: 'json',
      specifyBody: 'json',
      jsonBody: '={"name":"complexity_scorer","arguments":{"project_root":"{{ $json.project_root }}"}}'
    },
    continueOnFail: true
  },
  output: [{ skill: 'complexity_scorer', result: {} }]
});

const architectureValidator = node({
  type: 'n8n-nodes-base.httpRequest',
  version: 4.4,
  config: {
    name: 'Architecture Validator',
    position: [440, 480],
    parameters: {
      method: 'POST',
      url: 'http://localhost:8002/call',
      sendHeaders: true,
      specifyHeaders: 'keypair',
      headerParameters: {
        parameters: [
          { name: 'Authorization', value: '=("Bearer " + $env.MCP_API_TOKEN)' }
        ]
      },
      sendBody: true,
      contentType: 'json',
      specifyBody: 'json',
      jsonBody: '={"name":"architecture_validator","arguments":{"project_root":"{{ $json.project_root }}"}}'
    },
    continueOnFail: true
  },
  output: [{ skill: 'architecture_validator', result: {} }]
});

const combineResults = merge({
  version: 3.2,
  config: {
    name: 'Combine All Results',
    position: [680, 300],
    parameters: { mode: 'append' }
  }
});

const aggregateOutput = node({
  type: 'n8n-nodes-base.set',
  version: 3.4,
  config: {
    name: 'Aggregate Output',
    position: [900, 300],
    parameters: {
      mode: 'raw',
      jsonOutput: '={ "skills_results": $input.all().map(i => i.json), "total_skills_run": $input.all().length }'
    }
  },
  output: [{ skills_results: [], total_skills_run: 4 }]
});

export default workflow('aura-parallel-skills', 'AURA Parallel Skills Runner')
  .add(startTrigger)
  .to(setProjectRoot)
  .to(securityScanner.to(combineResults.input(0)))
  .add(setProjectRoot)
  .to(testCoverageAnalyzer.to(combineResults.input(1)))
  .add(setProjectRoot)
  .to(complexityScorer.to(combineResults.input(2)))
  .add(setProjectRoot)
  .to(architectureValidator.to(combineResults.input(3)))
  .add(combineResults)
  .to(aggregateOutput);
