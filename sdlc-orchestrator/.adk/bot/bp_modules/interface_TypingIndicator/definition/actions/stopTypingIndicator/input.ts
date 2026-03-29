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
          "The message ID from which the typing indicator should be removed",
        ),
    })
    .catchall(z.never()),
};
