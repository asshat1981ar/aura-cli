import { BotDefinition, z } from "@botpress/sdk";
    import {
    BUILT_IN_STATES,
    BUILT_IN_TAGS,
    TranscriptSchema,
    TrackedStateSchema,
    WorkflowCallbackEvent,
    WorkflowScheduleEvent,
    WorkflowContinueEvent,
    SubworkflowFinished,
    WorkflowDataRequestEvent,
  } from "@botpress/runtime/definition";
    
import integration_webchat from "./bp_modules/integration_webchat";

    const bot = new BotDefinition({

      attributes: {
        runtime: "adk",
        runtimeVersion: "1.16.7",
        
      },
      
configuration: {
schema: z.object({
}).catchall(z.never())
},

      user: {
        tags: {
          ...BUILT_IN_TAGS.user,
          
        },
      },
      message: {
        tags: {
          ...BUILT_IN_TAGS.message,
          
        },
      },
      conversation: {
        tags: {
          ...BUILT_IN_TAGS.conversation,
          // Project ID
          "projectId": {"title":"Project ID"},
          // Current Phase
          "phase": {"title":"Current Phase"},
        },
      },
      workflows: {
        "builtin_knowledge_indexing": {
        title: "builtin_knowledge_indexing",
        description: "Built-in workflow to re-index all data sources in a knowledge base",
        input: { schema: z.object({
kbName: z.string(),
kbId: z.string(),
force: z.default(z.boolean().describe("Force re-indexing even if files haven\'t changed"), false),
}).catchall(z.never()) },
        output: { schema: z.object({
processed: z.default(z.number(), 0),
added: z.default(z.array(z.object({
file: z.string(),
name: z.string(),
hash: z.string(),
size: z.number(),
}).catchall(z.never())), []),
updated: z.default(z.array(z.object({
file: z.string(),
name: z.string(),
hash: z.string(),
size: z.number(),
}).catchall(z.never())), []),
deleted: z.default(z.array(z.object({
file: z.string(),
name: z.string(),
hash: z.string(),
size: z.number(),
}).catchall(z.never())), []),
errors: z.default(z.array(z.string()), []),
}).catchall(z.never()) },
        tags: {
        ...BUILT_IN_TAGS.workflow,
        key: {
          title: "Workflow Key",
          description: "Unique key for workflow deduplication"
        },
      }
      }
      },
      actions: {
        "tablesRecomputeRows": {
          input: { schema: z.object({
tableId: z.string(),
botId: z.string(),
schema: z.any(),
requests: z.array(z.object({
row: z.record(z.string(), z.any()),
columnsToRecompute: z.array(z.string()),
}).catchall(z.never())),
}).catchall(z.never()) },
          output: { schema: z.object({
isFinished: z.boolean(),
rows: z.array(z.any()),
}).catchall(z.never()) }
        }
      },
      

      events: {
        [WorkflowScheduleEvent.name]: {
          schema: WorkflowScheduleEvent.schema,
        },
        [WorkflowCallbackEvent.name]: {
          schema: WorkflowCallbackEvent.schema,
        },
        [WorkflowContinueEvent.name]: {
          schema: WorkflowContinueEvent.schema,
        },
        [SubworkflowFinished.name]: {
          schema: SubworkflowFinished.schema,
        },
        [WorkflowDataRequestEvent.name]: {
          schema: WorkflowDataRequestEvent.schema,
        },
        
      },

      states: {
        /**
         * This is the ADK-native conversation state that contains the
         * necessary data to run the conversation and its handlers.
        */
        conversation: {
          type: "conversation",
          schema: z.object({ transcript: TranscriptSchema }),
        },

        /**
         * This is a generic state to store the conversation-specific state.
         * This is defined by the users at build-time when they define conversations.
         * Because each conversation can have its own state schema, we use `z.any()`
        */
        [BUILT_IN_STATES.conversation]: {
          type: "conversation",
          schema: TrackedStateSchema,
        },

        /**
         * Bot-wide global state that persists across all conversations
         */
        [BUILT_IN_STATES.bot]: {
          type: "bot",
          schema: TrackedStateSchema,
        },

        /**
         * User-specific state that persists across conversations for each user
         */
        [BUILT_IN_STATES.user]: {
          type: "user",
          schema: TrackedStateSchema,
        },

        /**
         * Workflow-specific state that persists across workflow executions
         */
        [BUILT_IN_STATES.workflowState]: {
          type: "workflow",
          schema: TrackedStateSchema,
        },

        /**
         * Workflow cached steps executions
         */
        [BUILT_IN_STATES.workflowSteps]: {
          type: "workflow",
          schema: TrackedStateSchema,
        },

        /**
         * Data source metadata for dashboard visibility (knowledge base sources)
         */
        [BUILT_IN_STATES.dsData]: {
          type: "bot",
          schema: z.record(z.any()),
        },
      },
    });
bot.addIntegration(integration_webchat, { alias: "webchat", enabled: true, configuration: {"allowedOrigins":[]} });

export default bot;