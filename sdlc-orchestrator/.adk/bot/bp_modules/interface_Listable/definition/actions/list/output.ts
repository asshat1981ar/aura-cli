/* eslint-disable */
/* tslint:disable */
// This file is generated. Do not edit it manually.

import { z } from "@botpress/sdk";
export const output = {
  schema: z
    .object({
      items: z.array(
        z.intersection(
          z.ref("item"),
          z.object({
            id: z.string(),
          }),
        ),
      ),
      meta: z
        .object({
          nextToken: z.optional(z.string()),
        })
        .catchall(z.never()),
    })
    .catchall(z.never()),
};
