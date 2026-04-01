/* eslint-disable */
/* tslint:disable */
// This file is generated. Do not edit it manually.

import { z } from "@botpress/sdk";
export const configuration = {
  schema: z
    .object({
      encryptionKey: z
        .optional(
          z
            .string()
            .title("Encryption Key (HS256) - optional")
            .describe(
              "Only set this config if you plan on signing your user key yourself. Key used to sign and verify user keys; JWT tokens with HS256 algorithm.",
            ),
        )
        .describe(
          "Only set this config if you plan on signing your user key yourself. Key used to sign and verify user keys; JWT tokens with HS256 algorithm.",
        ),
      webhookUrl: z
        .optional(
          z
            .string()
            .title("Webhook URL - optional")
            .describe(
              "Only set this config if you want to listen on a webhook instead of the standard SSE stream. URL where all incoming and outgoing messages / events are sent to.",
            ),
        )
        .describe(
          "Only set this config if you want to listen on a webhook instead of the standard SSE stream. URL where all incoming and outgoing messages / events are sent to.",
        ),
      webhookSecret: z
        .optional(
          z
            .string()
            .title("Webhook Secret - optional")
            .describe("Secret forwarded to the webhook URL."),
        )
        .describe("Secret forwarded to the webhook URL."),
    })
    .catchall(z.never()),
};
