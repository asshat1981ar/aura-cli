// Copyright (c) "Neo4j"
// Neo4j Sweden AB [http://neo4j.com]

package clierr

import (
	"errors"
	"fmt"
)

// UsageError indicates a user error requiring feedback (e.g., invalid input).
type UsageError struct {
	msg string
	err error
}

func (e *UsageError) Error() string { return e.msg }
func (e *UsageError) Unwrap() error { return e.err }

// NewUsageError creates an error indicating incorrect usage that requires user feedback.
func NewUsageError(msg string, a ...any) error {
	e := fmt.Errorf(msg, a...)
	return &UsageError{msg: e.Error(), err: errors.Unwrap(e)}
}

// UpstreamError indicates an API error that may be resolved by retrying.
type UpstreamError struct {
	msg string
	err error
}

func (e *UpstreamError) Error() string { return e.msg }
func (e *UpstreamError) Unwrap() error { return e.err }

// NewUpstreamError creates an error for API failures where a retry may resolve the issue.
func NewUpstreamError(msg string, a ...any) error {
	e := fmt.Errorf(msg, a...)
	return &UpstreamError{msg: e.Error(), err: errors.Unwrap(e)}
}

// FatalError indicates an unrecoverable error.
type FatalError struct {
	msg string
	err error
}

func (e *FatalError) Error() string { return e.msg }
func (e *FatalError) Unwrap() error { return e.err }

// NewFatalError creates an unrecoverable error.
func NewFatalError(msg string, a ...any) error {
	e := fmt.Errorf(msg, a...)
	return &FatalError{msg: e.Error(), err: errors.Unwrap(e)}
}
