// Copyright (c) "Neo4j"
// Neo4j Sweden AB [http://neo4j.com]

package utils

import (
	"github.com/neo4j/cli/common/clicfg"
	"github.com/spf13/cobra"
)

const (
	ORGANIZATION_ID_FLAG = "organization-id"
	PROJECT_ID_FLAG      = "project-id"
)

// This function is meant to run in the PreRun of V2 commands to ensure that the flags are marked as required if no values have been set
// through the `config project add/use` commands.
func SetProjectFlagsAsRequired(cfg *clicfg.Config, cmd *cobra.Command) error {
	defaultProject, err := cfg.Aura.Projects.Default()
	if err != nil {
		return err
	}
	if defaultProject.OrganizationId == "" {
		err := cmd.MarkFlagRequired(ORGANIZATION_ID_FLAG)
		if err != nil {
			return err
		}
	}

	if defaultProject.ProjectId == "" {
		err := cmd.MarkFlagRequired(PROJECT_ID_FLAG)
		if err != nil {
			return err
		}
	}

	return nil
}

// This function is meant to run in the RunE of V2 commands to ensure that the values are set as the given default values if no values are
// given via flags when running the command.
func SetProjectDefaults(cfg *clicfg.Config, organizationId string, projectId string) (string, string, error) {
	defaultProject, err := cfg.Aura.Projects.Default()
	if err != nil {
		return "", "", err
	}
	if organizationId == "" {
		organizationId = defaultProject.OrganizationId
	}
	if projectId == "" {
		projectId = defaultProject.ProjectId
	}
	return organizationId, projectId, err
}
