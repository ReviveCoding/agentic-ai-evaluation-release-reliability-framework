# GitHub Settings Checklist

## General

- [ ] Set repository description and topics
- [ ] Enable Issues
- [ ] Enable Discussions only if public Q&A will be maintained
- [ ] Disable Wiki unless it will be maintained
- [ ] Enable release immutability when available

## Security

- [ ] Enable Dependabot alerts
- [ ] Enable Dependabot security updates
- [ ] Enable private vulnerability reporting
- [ ] Enable secret scanning and push protection when available
- [ ] Confirm CodeQL completes successfully

## Main branch ruleset

- [ ] Require pull requests
- [ ] Require one approving review
- [ ] Require conversation resolution
- [ ] Require status checks
- [ ] Require branch to be up to date
- [ ] Block force pushes and branch deletion

Recommended checks: CI matrix jobs, package, Docker, Monte Carlo smoke, CodeQL,
and release-check.

## Release

- [ ] Push `main`
- [ ] Push annotated tag `v1.0.0`
- [ ] Create a GitHub Release from `v1.0.0`
- [ ] Verify generated release notes and public claim language
