/* eslint-disable */
/* tslint:disable */
// This file is generated. Do not edit it manually.

import { z } from "@botpress/sdk";
export const input = {
  schema: z
    .object({
      conversationId: z
        .string()
        .describe(
          "The conversation id of the webchat instance. Usually {{event.conversationId}}",
        ),
      config: z
        .string()
        .describe(
          "A JSON string representing the new configuration. You can use {{JSON.stringify(workflow.someVariable)}} to convert an object to JSON",
        ),
    })
    .catchall(z.never()),
};
