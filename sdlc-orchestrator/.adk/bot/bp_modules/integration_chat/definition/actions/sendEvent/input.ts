/* eslint-disable */
/* tslint:disable */
// This file is generated. Do not edit it manually.

import { z } from "@botpress/sdk";
export const input = {
  schema: z
    .object({
      conversationId: z
        .string()
        .title("Conversation ID")
        .describe("The ID of the conversation to send the event to"),
      payload: z
        .record(z.string(), z.any())
        .title("Payload")
        .describe("Custom data payload to send with the event"),
    })
    .catchall(z.never()),
};
