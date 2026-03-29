/* eslint-disable */
/* tslint:disable */
// This file is generated. Do not edit it manually.

import { z } from "@botpress/sdk";
export const input = {
  schema: z
    .object({
      userId: z
        .string()
        .describe(
          "The ID of the user. Usually you can access it using {{event.userId}}",
        ),
    })
    .catchall(z.never()),
};
