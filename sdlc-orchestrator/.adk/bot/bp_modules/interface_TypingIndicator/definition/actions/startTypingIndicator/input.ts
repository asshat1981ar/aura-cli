/* eslint-disable */
/* tslint:disable */
// This file is generated. Do not edit it manually.

import { z } from "@botpress/sdk";
export const input = {
  schema: z
    .object({
      conversationId: z.string(),
      messageId: z
        .string()
        .describe(
          "The message ID to which the typing indicator should be attached",
        ),
      timeout: z
        .optional(
          z
            .number()
            .describe(
              "The timeout in milliseconds after which the typing indicator should stop",
            ),
        )
        .describe(
          "The timeout in milliseconds after which the typing indicator should stop",
        ),
    })
    .catchall(z.never()),
};
