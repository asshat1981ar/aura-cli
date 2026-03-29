/* eslint-disable */
/* tslint:disable */
// This file is generated. Do not edit it manually.

import { z } from "@botpress/sdk";
export const bloc = {
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
        z.union([
          z
            .object({
              type: z.literal("text"),
              payload: z
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
                  value: z.optional(z.string()),
                })
                .catchall(z.never()),
            })
            .catchall(z.never()),
          z
            .object({
              type: z.literal("markdown"),
              payload: z
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
                  markdown: z.string().min(1, undefined),
                })
                .catchall(z.never()),
            })
            .catchall(z.never()),
          z
            .object({
              type: z.literal("image"),
              payload: z
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
                  imageUrl: z.string().min(1, undefined),
                })
                .catchall(z.never()),
            })
            .catchall(z.never()),
          z
            .object({
              type: z.literal("audio"),
              payload: z
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
                  audioUrl: z.string().min(1, undefined),
                })
                .catchall(z.never()),
            })
            .catchall(z.never()),
          z
            .object({
              type: z.literal("video"),
              payload: z
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
                  videoUrl: z.string().min(1, undefined),
                })
                .catchall(z.never()),
            })
            .catchall(z.never()),
          z
            .object({
              type: z.literal("file"),
              payload: z
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
                  fileUrl: z.string().min(1, undefined),
                  title: z.optional(z.string().min(1, undefined)),
                })
                .catchall(z.never()),
            })
            .catchall(z.never()),
          z
            .object({
              type: z.literal("location"),
              payload: z
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
                  latitude: z.number(),
                  longitude: z.number(),
                  address: z.optional(z.string()),
                  title: z.optional(z.string()),
                })
                .catchall(z.never()),
            })
            .catchall(z.never()),
        ]),
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
