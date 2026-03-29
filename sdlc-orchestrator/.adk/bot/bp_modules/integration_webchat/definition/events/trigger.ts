/* eslint-disable */
/* tslint:disable */
// This file is generated. Do not edit it manually.

import { z } from "@botpress/sdk";
export const trigger = {
  title: "Custom Trigger (advanced)",
  description:
    "This event occurs when a payload is sent from the browser. That payload will be available in {{event.payload}}. Usage: \n\nwindow.botpressWebChat.sendPayload({ type: 'trigger', payload: {} })",
  attributes: {},
  schema: z
    .object({
      origin: z.literal("website").describe("The origin of the event trigger"),
      userId: z
        .string()
        .describe("The webchat userId that triggered the event"),
      conversationId: z
        .string()
        .describe("The webchat conversationId that triggered the event"),
      payload: z
        .record(z.string(), z.any())
        .describe("The payload to send with the event"),
    })
    .catchall(z.never()),
};
