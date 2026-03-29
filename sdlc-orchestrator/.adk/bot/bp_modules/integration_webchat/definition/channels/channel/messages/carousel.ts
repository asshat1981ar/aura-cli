/* eslint-disable */
/* tslint:disable */
// This file is generated. Do not edit it manually.

import { z } from "@botpress/sdk";
export const carousel = {
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
      items: z.array(
        z
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
            title: z.string().min(1, undefined),
            subtitle: z.optional(z.string().min(1, undefined)),
            imageUrl: z.optional(z.string().min(1, undefined)),
            actions: z.array(
              z
                .object({
                  action: z.enum(["postback", "url", "say"]),
                  label: z.string().min(1, undefined),
                  value: z.string().min(1, undefined),
                })
                .catchall(z.never()),
            ),
          })
          .catchall(z.never()),
      ),
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
