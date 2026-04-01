const myTrigger = {
  type: "n8n-nodes-base.manualTrigger",
  name: "Manual Trigger",
  version: 1,
  position: [0, 0],
  parameters: {}
};

const myWorkflow = workflow({
  nodes: [myTrigger],
  connections: {}
});

export default myWorkflow;
