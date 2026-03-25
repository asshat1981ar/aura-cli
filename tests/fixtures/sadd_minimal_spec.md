# Fix Authentication Bug

Simple single-task spec for testing minimal parsing.

## Task: Patch Token Validation

Fix the JWT token validation in the auth middleware.

- Update token expiry check
- Add refresh token support
- Acceptance: auth tests pass
