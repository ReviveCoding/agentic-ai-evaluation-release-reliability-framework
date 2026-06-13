# Threat Model

## Scope

This covers the local evaluation framework, synthetic tool runtime, retrieval,
trace store, replay, reports, and CI. It does not claim production isolation or
authorization for real-world tools.

## Assets

- model and dataset fingerprints
- evaluation scenarios and expected trajectories
- retrieval evidence and provenance
- replay signatures and release decisions
- configuration thresholds
- CI credentials and repository integrity

## Primary threats and mitigations

### Prompt or context injection

Mitigations include a structured tool registry, forbidden-tool evaluation,
self-confirmation conflict guard, and PASS/REVIEW/BLOCK policy.

### Retrieval poisoning or stale-context dominance

Mitigations include recency-aware queries, service/tool compatibility,
strict/compatible provenance metrics, and deterministic conflict tests.

### Unsafe or premature action

Mitigations include slot validation, clarification policy, terminal-safety
evaluation, and unsafe-completion/false-block metrics.

### Replay tampering or false reproducibility

Mitigations include fingerprints, stable-field signatures, explicit volatile
field exclusion, and replay regression tests.

### Sensitive-data leakage

Mitigations include `.gitignore`, a public-release audit for secrets and large
files, and only tiny public smoke fixtures.

### CI supply-chain compromise

Mitigations include full-SHA Action pinning, least-privilege permissions,
Dependabot, CodeQL, and protected-branch checks after publication.

## Residual risks

Lexical and learned retrieval can fail on unseen phrasing. Same-environment
replay does not guarantee cross-platform bitwise identity. Synthetic safety
behavior does not replace production authorization controls.
