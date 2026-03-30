/* eslint-disable */
/* tslint:disable */
// This file is generated. Do not edit it manually.

import { z } from "@botpress/sdk";
export const userAugmentedData = {
  type: "user" as const,
  schema: z
    .object({
      ip: z.optional(z.string()),
      location: z.optional(
        z
          .object({
            city: z.optional(z.string()),
            region: z.optional(z.string()),
            country: z.optional(z.string()),
          })
          .catchall(z.never()),
      ),
      device: z.optional(
        z
          .object({
            os: z.optional(z.string()),
            browser: z.optional(z.string()),
            browser_version: z.optional(z.string()),
            type: z.optional(z.string()),
          })
          .catchall(z.never()),
      ),
    })
    .catchall(z.never()),
};
