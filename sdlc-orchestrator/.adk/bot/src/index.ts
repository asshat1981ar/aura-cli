import * as bp from '.botpress'
import { setupAdkRuntime } from './adk-runtime'
import { Actions } from './actions'
import {isMainThread, isWorkerMode, initializeParentWorker, runWorker, BuiltInActions} from '@botpress/runtime/internal'
import { handlers } from "@botpress/runtime/runtime";

const bot = new bp.Bot({
  actions: {},
  register: async (props) => {
    handlers.trigger.triggerRegisterEvent(props)
  }
})


// ============================================================================
// WORKER INITIALIZATION
// ============================================================================

if (isWorkerMode() && isMainThread) {
  // Branch 1: Main thread in worker mode - initialize parent with pool
  if (process.env.BP_DEBUG) console.log("[Main] Initializing parent worker with pool...");
  initializeParentWorker(bot);
} else if (isWorkerMode() && process.env.IS_DEV_WORKER === "true") {
  // Branch 2: Worker thread - run child worker
  if (process.env.BP_DEBUG) console.log("[Worker] Initializing child worker...");
  runWorker(bot);
  setupAdkRuntime(bot);
} else {
  // Branch 3: Worker mode disabled - single-thread mode
  if (process.env.BP_DEBUG) console.log("[Bot] Running in single-thread mode");
  setupAdkRuntime(bot);
}

export default bot