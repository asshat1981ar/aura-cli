import { InterfacePackage } from "@botpress/sdk";


import interface_TypingIndicator from "../bp_modules/interface_TypingIndicator";
import interface_Llm from "../bp_modules/interface_Llm";
import interface_Listable from "../bp_modules/interface_Listable";

export const InterfaceDefinitions = {
 TypingIndicator: interface_TypingIndicator,
Llm: interface_Llm,
Listable: interface_Listable
} as Record<string, InterfacePackage>;