/* eslint-disable */
/* tslint:disable */
// This file is generated. Do not edit it manually.

import { z } from "@botpress/sdk";
export const choice = {
  schema: z
    .object({
      userId: z
        .optional(
          z
            .string()
            .describe(
              "Fake User ID; allows the bot to pretend sending the message as someone else.",
            ),
        )
        .describe(
          "Fake User ID; allows the bot to pretend sending the message as someone else.",
        ),
      text: z.string().min(1, undefined),
      options: z.array(
        z
          .object({
            label: z.string().min(1, undefined),
            value: z.string().min(1, undefined),
          })
          .catchall(z.never()),
      ),
      disableFreeText: z.optional(z.boolean()),
      type: z
        .optional(
          z
            .string()
            .describe(
              "Type of the message; same as the top level message type",
            ),
        )
        .describe("Type of the message; same as the top level message type"),
      metadata: z.optional(z.record(z.string(), z.any())),
    })
    .catchall(z.never()),
};
