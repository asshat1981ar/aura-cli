import { workflow, node, trigger, ifElse } from '@n8n/workflow-sdk';

const webhookTrigger = trigger({
  type: 'n8n-nodes-base.webhook',
  version: 2.1,
  config: {
    name: 'AURA Notify Webhook',
    position: [0, 300],
    parameters: {
      httpMethod: 'POST',
      path: 'aura-notify',
      responseMode: 'responseNode'
    }
  },
  output: [{ body: { verification_status: 'pass', goal: 'example', cycle: 1 } }]
});

const checkStatus = ifElse({
  version: 2.3,
  config: {
    name: 'Check Verification Status',
    position: [240, 300],
    parameters: {
      conditions: {
        conditions: [
          {
            leftValue: '={{ $json.body.verification_status }}',
            operator: { type: 'string', operation: 'equals' },
            rightValue: 'fail'
          }
        ]
      },
      looseTypeValidation: true
    }
  }
});

const formatFailure = node({
  type: 'n8n-nodes-base.set',
  version: 3.4,
  config: {
    name: 'Format Failure Alert',
    position: [480, 180],
    parameters: {
      mode: 'raw',
      jsonOutput: '={ "status": "FAIL", "message": "AURA cycle failed verification", "goal": $json.body.goal, "cycle": $json.body.cycle, "learnings": $json.body.learnings, "timestamp": new Date().toISOString() }'
    },
    continueOnFail: true
  },
  output: [{ status: 'FAIL', message: 'AURA cycle failed verification' }]
});

const formatSuccess = node({
  type: 'n8n-nodes-base.set',
  version: 3.4,
  config: {
    name: 'Format Success Message',
    position: [480, 420],
    parameters: {
      mode: 'raw',
      jsonOutput: '={ "status": "PASS", "message": "AURA cycle completed successfully", "goal": $json.body.goal, "cycle": $json.body.cycle, "timestamp": new Date().toISOString() }'
    },
    continueOnFail: true
  },
  output: [{ status: 'PASS', message: 'AURA cycle completed successfully' }]
});

const respondToWebhook = node({
  type: 'n8n-nodes-base.respondToWebhook',
  version: 1.1,
  config: {
    name: 'Respond to Webhook',
    position: [720, 300],
    parameters: {
      respondWith: 'json',
      responseBody: '={ { "received": true, "status": $json.status, "message": $json.message } }'
    }
  },
  output: [{ received: true }]
});

export default workflow('aura-notification-hub', 'AURA Notification Hub')
  .add(webhookTrigger)
  .to(checkStatus
    .onTrue(formatFailure.to(respondToWebhook))
    .onFalse(formatSuccess.to(respondToWebhook))
  );
