import * as bp from ".botpress";
import { IntegrationDefinitions } from "./integrations";
import { InterfaceDefinitions } from "./interfaces";
import { initializeAssets } from "./assets-runtime";
import { handlers, patchHandlers, agentRegistry, z, initialize, register, registerIntegration } from "@botpress/runtime/runtime";
import { AgentConfig } from "./config";
import { Conversations } from "./conversations";
import { Knowledge } from "./knowledge";
import { Triggers } from "./triggers";
import { Workflows } from "./workflows";
import { Actions } from "./actions";
import { Tables } from "./tables";
import { Interfaces } from "../../interfaces";

export function setupAdkRuntime(bot: bp.Bot) {
  // Initialize global error handlers for the runtime

  initialize({ config: AgentConfig })

  register(...Object.values(Conversations));
  register(...Object.values(Workflows));
  register(...Object.values(Triggers));
  register(...Object.values(Actions));
  register(...Object.values(Tables));
  register(...Object.values(Knowledge));

  for (const [alias, { definition }] of Object.entries(IntegrationDefinitions)) {
    registerIntegration({ alias, definition });
  }

  // Initialize the global agent registry
  agentRegistry.initialize({
    integrations: Object.entries(IntegrationDefinitions).map(([alias, def]) => ({ ...def, alias })),
    interfaces: Object.entries(InterfaceDefinitions).map(([alias, def]) => ({ ...def, alias })),
    interfacesMapping: Interfaces as Record<string, any>
  });

  // Patch bot handlers to add runtime context
  patchHandlers(bot);

  // Initialize assets system
  initializeAssets();

  // Setup conversation, trigger, and workflow handlers
  handlers.conversation.setup(bot);
  handlers.event.setup(bot);
  handlers.trigger.setup(bot);
  handlers.workflow.setup(bot);
  handlers.actions.setup(bot);
  handlers.plugins.setup(bot);
}