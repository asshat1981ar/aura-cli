/* eslint-disable */
/* tslint:disable */
// This file is generated. Do not edit it manually.

import * as sdk from "@botpress/sdk"

import * as configuration from "./configuration/index"
import * as configurations from "./configurations/index"
import * as actions from "./actions/index"
import * as channels from "./channels/index"
import * as events from "./events/index"
import * as states from "./states/index"
import * as entities from "./entities/index"
import * as interfaces from "./interfaces/index"
export * as configuration from "./configuration/index"
export * as configurations from "./configurations/index"
export * as actions from "./actions/index"
export * as channels from "./channels/index"
export * as events from "./events/index"
export * as states from "./states/index"
export * as entities from "./entities/index"
export * as interfaces from "./interfaces/index"

export default {
  name: "chat",
  version: "0.7.6",
  attributes: {},
  user: { "tags": { "fid": { "title": "Foreign ID", "description": "Copy of the foreign ID of the user" }, "profile": { "title": "User Profile", "description": "Custom profile data of the user encoded as a string" } }, "creation": { "enabled": false, "requiredTags": [] } },
  configuration: configuration.configuration,
  configurations: configurations.configurations,
  actions: actions.actions,
  channels: channels.channels,
  events: events.events,
  states: states.states,
  entities: entities.entities,
  interfaces: interfaces.interfaces,
} satisfies sdk.IntegrationPackage["definition"]