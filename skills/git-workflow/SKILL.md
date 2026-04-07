---
name: git-workflow
description: Use this skill when you need to perform git operations, code commits, or code reviews.
---

# Git Workflow Skill

Follow these strict git workflows:

## 1. Commit Rules
- **NEVER** amend published commits (commits already pushed to remote).
- **NEVER** run destructive commands like `git reset --hard`, `git push --force`, or `git clean -fd` unless explicitly instructed.
- Create meaningful commit messages: start with a verb (Add, Fix, Update, Remove).
- Keep commits atomic: one logical change per commit.

## 2. Before Committing
- Always run `git status` and `git diff` to review changes before committing.
- Never commit secrets, credentials, or sensitive data (check `.gitignore` first).

## 3. Branching
- Branch naming: `feature/`, `fix/`, `refactor/` prefixes.
- Main/protection branch (usually `main`) should never be force-pushed.

## 4. Merge Strategy
- Prefer creating a merge commit (not fast-forward) for feature branches.
- Use `git merge --no-ff` when merging long-lived feature branches.

## 5. Safe Operations
- `git stash` before switching branches with uncommitted changes.
- Always verify the target branch before merging.
