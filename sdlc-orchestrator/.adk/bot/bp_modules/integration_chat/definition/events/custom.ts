/* eslint-disable */
/* tslint:disable */
// This file is generated. Do not edit it manually.

import { z } from "@botpress/sdk";
export const custom = {
  title: "Custom Event",
  description: "Custom event sent from the chat client to the bot",
  attributes: {},
  schema: z
    .object({
      userId: z
        .string()
        .title("User ID")
        .describe("The ID of the user who sent the custom event"),
      conversationId: z
        .string()
        .title("Conversation ID")
        .describe("The ID of the conversation where the event was sent"),
      payload: z
        .record(z.string(), z.any())
        .title("Payload")
        .describe("Custom data payload sent with the event"),
    })
    .catchall(z.never()),
};
