import { workflow, node, trigger, ifElse } from '@n8n/workflow-sdk';

const startTrigger = trigger({
  type: 'n8n-nodes-base.manualTrigger',
  version: 1,
  config: {
    name: 'Goal Input',
    position: [0, 300]
  },
  output: [{ goal: 'Implement new feature', priority: 'high', dry_run: false }]
});

const classifyComplexity = node({
  type: 'n8n-nodes-base.code',
  version: 2,
  config: {
    name: 'Classify Complexity',
    position: [220, 300],
    parameters: {
      jsCode: `const goal = $input.first().json.goal.toLowerCase();
const highComplexity = ['implement','refactor','rewrite','add feature','create','build','fix bug'].some(k => goal.includes(k));
return [{ json: { ...($input.first().json), complexity: highComplexity ? 'high' : 'low' } }];`
    }
  },
  output: [{ goal: 'example', priority: 'high', dry_run: false, complexity: 'high' }]
});

const routeByComplexity = ifElse({
  version: 2.3,
  config: {
    name: 'Route by Complexity',
    position: [440, 300],
    parameters: {
      conditions: {
        conditions: [
          {
            leftValue: '={{ $json.complexity }}',
            operator: { type: 'string', operation: 'equals' },
            rightValue: 'high'
          }
        ]
      },
      looseTypeValidation: true
    }
  }
});

const callAuraPipeline = node({
  type: 'n8n-nodes-base.httpRequest',
  version: 4.4,
  config: {
    name: 'Call AURA Pipeline',
    position: [680, 160],
    parameters: {
      method: 'POST',
      url: 'http://localhost:8001/webhook/goal',
      sendHeaders: true,
      specifyHeaders: 'keypair',
      headerParameters: {
        parameters: [
          { name: 'Authorization', value: '=("Bearer " + $env.AGENT_API_TOKEN)' },
          { name: 'Content-Type', value: 'application/json' }
        ]
      },
      sendBody: true,
      contentType: 'json',
      specifyBody: 'json',
      jsonBody: '={"goal": "{{ $json.goal }}", "priority": "{{ $json.priority }}", "dry_run": {{ $json.dry_run }}}'
    },
    continueOnFail: true
  },
  output: [{ status: 'queued', goal_id: '' }]
});

const callAiAgent = node({
  type: 'n8n-nodes-base.executeWorkflow',
  version: 1.3,
  config: {
    name: 'Run AI Agent Workflow',
    position: [680, 440],
    parameters: {
      source: 'database',
      workflowId: { __rl: true, value: 'aura-ai-agent', mode: 'id' },
      options: {
        waitForSubWorkflow: true
      }
    },
    continueOnFail: true
  },
  output: [{ agent_result: '', goal: '' }]
});

const formatResponse = node({
  type: 'n8n-nodes-base.set',
  version: 3.4,
  config: {
    name: 'Format Final Response',
    position: [920, 300],
    parameters: {
      mode: 'raw',
      jsonOutput: '={ "status": "accepted", "goal": $("Goal Input").first().json.goal, "complexity": $("Classify Complexity").first().json.complexity, "route": $("Classify Complexity").first().json.complexity === "high" ? "aura_pipeline" : "ai_agent", "result": $json, "timestamp": new Date().toISOString() }'
    }
  },
  output: [{ status: 'accepted', goal: '', complexity: '', route: '', result: {} }]
});

export default workflow('aura-goal-orchestrator', 'AURA Goal Orchestrator')
  .add(startTrigger)
  .to(classifyComplexity)
  .to(routeByComplexity
    .onTrue(callAuraPipeline.to(formatResponse))
    .onFalse(callAiAgent.to(formatResponse))
  );
