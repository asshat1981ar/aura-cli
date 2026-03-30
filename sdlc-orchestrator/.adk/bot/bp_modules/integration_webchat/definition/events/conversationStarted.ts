/* eslint-disable */
/* tslint:disable */
// This file is generated. Do not edit it manually.

import { z } from "@botpress/sdk";
export const conversationStarted = {
  title: "Conversation Started",
  description:
    "This event occurs when a user activates the webchat widget, prompting the interface to appear.",
  attributes: {},
  schema: z
    .object({
      userId: z.string(),
      conversationId: z.string(),
    })
    .catchall(z.never()),
};
