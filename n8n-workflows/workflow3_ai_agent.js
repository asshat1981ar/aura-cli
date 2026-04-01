import { workflow, node, trigger, tool, languageModel } from '@n8n/workflow-sdk';

// AI model - using OpenAI via environment credentials
const openAiModel = languageModel({
  type: '@n8n/n8n-nodes-langchain.lmChatOpenAi',
  version: 1.3,
  config: {
    name: 'OpenAI GPT-4o',
    position: [440, 500],
    parameters: {
      model: '={{ "gpt-4o-mini" }}'
    }
  }
});

// Security scanner tool
const securityScanTool = tool({
  type: 'n8n-nodes-base.httpRequest',
  version: 4.4,
  config: {
    name: 'scan_security',
    position: [640, 500],
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
      jsonBody: '{"name":"security_scanner","arguments":{"project_root":"/home/westonaaron675/aura-cli"}}'
    },
    continueOnFail: true
  }
});

// Test coverage tool
const coverageTool = tool({
  type: 'n8n-nodes-base.httpRequest',
  version: 4.4,
  config: {
    name: 'check_coverage',
    position: [840, 500],
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
      jsonBody: '{"name":"test_coverage_analyzer","arguments":{"project_root":"/home/westonaaron675/aura-cli"}}'
    },
    continueOnFail: true
  }
});

const startTrigger = trigger({
  type: 'n8n-nodes-base.manualTrigger',
  version: 1,
  config: {
    name: 'Start with Goal',
    position: [0, 300]
  },
  output: [{ goal: 'Analyze the security and test coverage of the project' }]
});

const aiAgent = node({
  type: '@n8n/n8n-nodes-langchain.agent',
  version: 1.7,
  config: {
    name: 'AURA Skills Agent',
    position: [440, 300],
    parameters: {
      options: {
        systemMessage: 'You are an AURA code analysis agent. Use the available tools to analyze the project. When asked about security, use the scan_security tool. When asked about test coverage, use the check_coverage tool. Synthesize the results into a clear summary.',
        humanMessage: '={{ $json.goal || "Analyze the AURA project codebase for issues." }}'
      }
    },
    subnodes: {
      model: openAiModel,
      tools: [securityScanTool, coverageTool]
    },
    continueOnFail: true
  },
  output: [{ output: 'Analysis complete' }]
});

const formatResult = node({
  type: 'n8n-nodes-base.set',
  version: 3.4,
  config: {
    name: 'Format Agent Result',
    position: [700, 300],
    parameters: {
      mode: 'raw',
      jsonOutput: '={ "agent_result": $json.output, "goal": $("Start with Goal").first().json.goal, "timestamp": new Date().toISOString() }'
    }
  },
  output: [{ agent_result: '', goal: '' }]
});

export default workflow('aura-ai-agent', 'AURA n8n AI Agent')
  .add(startTrigger)
  .to(aiAgent)
  .to(formatResult);
