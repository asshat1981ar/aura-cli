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
      event: z
        .string()
        .describe("An event as JSON to send to the webchat instance"),
    })
    .catchall(z.never()),
};
