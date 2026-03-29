////////////////////////////////////////////////////////
// SDLC Orchestrator — Agent Configuration
// Imported by .adk/bot/src/config.ts (auto-generated ADK).
// Edit this file to customise bot-level settings.
////////////////////////////////////////////////////////

const agentConfig = {
  /** Human-readable bot name shown in the Botpress Studio. */
  name: "SDLC Orchestrator",

  /**
   * Conversation-level tags.
   * - projectId: links a conversation to a tracked software project
   * - phase:     current SDLC phase (ideation | planning | dev | qa | release | maintenance)
   */
  conversationTags: {
    projectId: { title: "Project ID" },
    phase: { title: "Current Phase" },
  },
};

export default agentConfig;
