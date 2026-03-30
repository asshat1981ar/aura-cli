/* eslint-disable */
/* tslint:disable */
// This file is generated. Do not edit it manually.

import { z } from "@botpress/sdk";
export const input = {
  schema: z
    .object({
      name: z.optional(z.string()),
      pictureUrl: z.optional(z.string()),
      email: z.optional(z.string()),
      user: z
        .object({
          id: z.string().describe("Some unique identifier for the user"),
          conversationId: z
            .string()
            .describe("The ID of the conversation in which to add the user"),
        })
        .catchall(z.never()),
    })
    .catchall(z.never()),
};
