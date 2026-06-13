# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 1.0.x | Yes |
| Earlier versions | No |

## Reporting a vulnerability

Do not disclose suspected vulnerabilities in a public issue.

Use GitHub's **Report a vulnerability** option in the Security tab when private
vulnerability reporting is enabled. If it is unavailable, open a minimal issue
asking the maintainer for a private reporting channel without including exploit
details, credentials, private data, or proof-of-concept payloads.

Include the affected version, impact, minimal reproduction steps, and a
suggested mitigation when available.

## Security boundaries

This repository is an evaluation framework, not a hardened production agent.
Synthetic tools do not authorize real-world actions. See
[`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) and
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).
