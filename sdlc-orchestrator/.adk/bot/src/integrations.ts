import { IntegrationPackage } from "@botpress/sdk";


import integration_webchat from "../bp_modules/integration_webchat";

export const IntegrationDefinitions = {
 "webchat": integration_webchat
} as Record<string, IntegrationPackage>;