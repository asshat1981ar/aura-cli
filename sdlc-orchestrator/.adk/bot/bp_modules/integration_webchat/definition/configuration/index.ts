/* eslint-disable */
/* tslint:disable */
// This file is generated. Do not edit it manually.

import { z } from "@botpress/sdk";
export const configuration = {
  schema: z
    .object({
      jwtPublicKey: z
        .optional(
          z
            .string()
            .describe(
              "The JWT public key used to verify tokens. When specified, userData and attributes in createUser and updateUser should be signed JWT tokens.",
            ),
        )
        .describe(
          "The JWT public key used to verify tokens. When specified, userData and attributes in createUser and updateUser should be signed JWT tokens.",
        ),
      adminSecret: z
        .optional(
          z.string().describe("The secret key required to manage users"),
        )
        .describe("The secret key required to manage users"),
      version: z.optional(z.string()),
      botDisplayName: z.optional(z.string()),
      botAvatarUrl: z.optional(z.string()),
      botDescription: z.optional(z.string()),
      fabImage: z.optional(z.string()),
      descriptionPhoneNumber: z.optional(
        z
          .object({
            title: z.optional(z.string()),
            link: z.optional(z.string()),
          })
          .catchall(z.never()),
      ),
      descriptionEmailAddress: z.optional(
        z
          .object({
            title: z.optional(z.string()),
            link: z.optional(z.string()),
          })
          .catchall(z.never()),
      ),
      descriptionWebsiteUrl: z.optional(
        z
          .object({
            title: z.optional(z.string()),
            link: z.optional(z.string()),
          })
          .catchall(z.never()),
      ),
      termsConditionsUrl: z.optional(
        z
          .object({
            title: z.optional(z.string()),
            link: z.optional(z.string()),
          })
          .catchall(z.never()),
      ),
      privacyPolicyUrl: z.optional(
        z
          .object({
            title: z.optional(z.string()),
            link: z.optional(z.string()),
          })
          .catchall(z.never()),
      ),
      botComposerPlaceholder: z.optional(z.string()),
      showPoweredByBotpress: z.optional(z.boolean()),
      additionalStylesheet: z.optional(z.string()),
      additionalStylesheetUrl: z.optional(z.string()),
      allowedOrigins: z.optional(z.array(z.string())),
      primaryColor: z.optional(z.string()),
      borderRadiusScale: z.optional(z.number()),
      themeMode: z.optional(z.enum(["light", "dark"])),
      variant: z.optional(z.enum(["solid", "soft"])),
      headerVariant: z.optional(z.enum(["solid", "glass"])),
      fontFamily: z.optional(z.string()),
      feedbackEnabled: z.optional(z.boolean()),
      showPoweredBy: z.optional(z.boolean()),
      footer: z.optional(z.string()),
      allowFileUpload: z.optional(z.boolean()),
      storageLocation: z.optional(z.enum(["sessionStorage", "localStorage"])),
      soundEnabled: z.optional(z.boolean()),
      toggleChatId: z.optional(z.string()),
      embeddedChatId: z.optional(z.string()),
      proactiveMessageEnabled: z.optional(z.boolean()),
      proactiveBubbleMessage: z.optional(z.string()),
      proactiveBubbleTriggerType: z.optional(
        z.enum(["afterDelay", "userIdle"]),
      ),
      proactiveBubbleDelayTime: z.optional(z.number()),
      conversationHistory: z.optional(z.boolean()),
      _debug: z.optional(z.boolean()),
    })
    .catchall(z.any()),
};
