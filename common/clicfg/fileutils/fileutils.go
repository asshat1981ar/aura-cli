// Copyright (c) "Neo4j"
// Neo4j Sweden AB [http://neo4j.com]

package fileutils

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/afero"
)

// ReadFileSafe reads a file; if it doesn't exist, returns empty []byte.
func ReadFileSafe(fs afero.Fs, path string) []byte {
	exists := FileExists(fs, path)

	if exists {
		data, err := afero.ReadFile(fs, path)
		if err != nil {
			panic(fmt.Sprintf("failed to read file %s: %v", path, err))
		}
		return data
	}
	return []byte{}
}

// ReadOrCreateFile reads a file or creates it if it doesn't exist.
func ReadOrCreateFile(fs afero.Fs, path string) []byte {
	exists := FileExists(fs, path)

	if exists {
		data, err := afero.ReadFile(fs, path)
		if err != nil {
			panic(fmt.Sprintf("failed to read file %s: %v", path, err))
		}
		return data
	}
	createFile(fs, path)
	return []byte{}
}

// WriteFile writes data to a file with secure permissions (0600).
func WriteFile(fs afero.Fs, path string, data []byte) {
	err := afero.WriteFile(fs, path, data, 0600)
	if err != nil {
		panic(fmt.Sprintf("failed to write file %s: %v", path, err))
	}
}

// FileExists checks whether a file exists at the given path.
func FileExists(fs afero.Fs, path string) bool {
	if _, err := fs.Stat(path); err == nil {
		return true
	} else if errors.Is(err, os.ErrNotExist) {
		return false
	} else {
		panic(fmt.Sprintf("failed to check file existence %s: %v", path, err))
	}
}

func createFile(fs afero.Fs, path string) {
	if err := fs.MkdirAll(filepath.Dir(path), 0755); err != nil {
		panic(fmt.Sprintf("failed to create directory for %s: %v", path, err))
	}

	_, err := fs.Create(path)
	if err != nil {
		panic(fmt.Sprintf("failed to create file %s: %v", path, err))
	}

	if err = fs.Chmod(path, 0600); err != nil {
		panic(fmt.Sprintf("failed to set permissions on %s: %v", path, err))
	}
}
