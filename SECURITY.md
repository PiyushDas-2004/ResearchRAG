# Security Policy

## Supported versions

Security fixes are made on the latest release of the `main` branch.

## Reporting a vulnerability

Please **do not open a public issue**. Use GitHub's private vulnerability reporting
("Security" tab -> "Report a vulnerability") on this repository. Include steps to reproduce,
affected versions and impact. You should receive an acknowledgement within 7 days.

Prompt-injection bypasses (a document that makes the system follow its instructions, leak the
system prompt, or emit uncited content) are in scope; please include a synthetic PDF or text
that reproduces it, never private documents.

See [docs/SECURITY.md](docs/SECURITY.md) for the threat model and the defences in place.
