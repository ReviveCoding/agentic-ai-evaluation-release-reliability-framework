# Release Checklist

## Functional freeze

- [x] GPU tool-policy training validated
- [x] Dialogue-group ID split and official OOD evaluation validated
- [x] Retrieval hardening and self-confirmation guard validated
- [x] Stable-field replay validated at 100.00%
- [x] PASS/REVIEW/BLOCK release gates validated
- [x] 70 regression tests passed
- [x] Project audit passed
- [x] Dependency check passed

## Documentation

- [x] README
- [x] Architecture
- [x] Results
- [x] Model card
- [x] Limitations
- [x] Reproducibility
- [x] Resume claims
- [x] Changelog
- [x] Release manifest

## Git commands to run manually

```powershell
git status
git add README.md PROJECT_SUMMARY.md CHANGELOG.md CITATION.cff VERSION docs reports/final_release_summary.md reports/final_release_manifest.json pyproject.toml src/agentic_eval_framework/__init__.py .gitignore
git commit -m "docs: finalize v1.0.0 release documentation"
git tag -a v1.0.0 -m "Agentic AI Evaluation & Release Reliability Framework v1.0.0"
git push origin HEAD
git push origin v1.0.0
```

Review `git diff --cached` before committing. The documentation script does not commit or push automatically.
