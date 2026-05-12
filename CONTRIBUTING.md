# Contributing Guide — AI-Powered Educational Support System

## Branch Strategy

This project uses a **trunk-based development** workflow with three levels:

```
main          ← stable, production-ready releases only
  └── dev     ← integration branch, unstable but functional
        └── feat/featureName   ← all active development happens here
```

### Branch Rules

| Branch | Purpose | Who merges into it | Direct commits |
|---|---|---|---|
| `main` | Stable releases | PR from `dev` only | ❌ Never |
| `dev` | Integration / unstable | PR from `feat/*` | ❌ Never |
| `feat/*` | Feature development | — | ✅ Yes |

---

## Starting a New Feature

Always branch off `dev`, never off `main`.

```bash
# Make sure your local dev is up to date
git checkout dev
git pull origin dev

# Create your feature branch
git checkout -b feat/datasetGenerator
```

### Branch naming convention

```
feat/datasetGenerator       ← new feature
feat/ragPipeline
feat/tutorAgent
fix/driveAuthError          ← bug fix
chore/updateDependencies    ← maintenance, no functional change
docs/ragArchitecture        ← documentation only
```

---

## Daily Development Loop

```bash
# Work on your feature
git add .
git commit -m "feat: add recursive Drive folder traversal"

# Push to remote regularly — don't let work sit only on your machine
git push origin feat/datasetGenerator
```

### Commit message format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short description>

Types:
  feat     → new feature or capability
  fix      → bug fix
  chore    → dependency update, config change, tooling
  docs     → documentation only
  refactor → code change with no behaviour change
  test     → adding or updating tests
```

Examples:
```
feat: add MarkItDown converter for PPTX and DOCX files
fix: handle empty chunks from scanned PDFs gracefully
chore: add google-auth to requirements.txt
refactor: extract QA generation into separate function
```

---

## Merging a Feature into Dev

When your feature is complete and tested locally:

```bash

#  Open a Pull Request: feat/datasetGenerator → dev
#    On GitHub: Compare & pull request
#    Title: same as your main commit message
#    Description: what it does, how to test it
```

Once the PR is reviewed and approved → **Squash and merge** into `dev`.
Delete the feature branch after merging.

---

## Merging Dev into Main (stable release)

Only done when `dev` is stable and a meaningful milestone is reached
(e.g. end of Phase 2, working RAG pipeline, first agent deployed).

```bash
# 1. Make sure dev is stable and all tests pass
git checkout dev
git pull origin dev

# 2. Open a Pull Request: dev → main
#    Title: "release: Phase 2 — dataset generation complete"
#    Description: summary of what's included in this release

# 3. Merge (no squash — preserve history from dev)
```

Tag the release after merging:
```bash
git checkout main
git pull origin main
git tag -a v0.2.0 -m "Phase 2 complete — dataset generator"
git push origin v0.2.0
```

### Version naming

```
v0.1.0  → Phase 1 complete (architecture, literature review)
v0.2.0  → Phase 2 complete (dataset generation, RAG pipeline)
v0.3.0  → Phase 3 complete (knowledge base, hybrid retrieval)
...and so on per phase
```

---

## Visual Summary

```
main     ─────────────────────────────────●─────────────────────●
                                          ↑                     ↑
                                     v0.1.0                v0.2.0
                                          │
dev      ──────────────●────────●─────────●──────●───────────────
                       ↑        ↑                ↑
                  feat/A      feat/B          feat/C
                  merged      merged          merged

feat/A   ──●──●──●──┘
feat/B            ──●──●──●──┘
feat/C                           ──●──●──●──┘
```

---

## Rules Summary

- **Never commit directly to `main` or `dev`**
- **Always branch from `dev`**, not from `main`
- **Delete feature branches** after merging
- **Never commit secrets** — `.env` and `service_account.json` are gitignored
- **Keep commits atomic** — one logical change per commit