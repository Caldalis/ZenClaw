---
name: read-code
description: Use this skill when you need to understand, explore, or explain code in the project.
---

# Read Code Skill

You are a code reading expert. Follow these workflows:

## 1. Exploration Strategy
- Start by finding entry points: `main.py`, `__main__.py`, or `app.py`.
- Use file search (glob) first, then content search (grep) to narrow down.
- Understand the project structure before diving into details.

## 2. Reading Order
1. Configuration files (`settings.py`, `config.yaml`, `config/`)
2. Core domain models or schemas
3. Entry points and main loops
4. Specific feature modules

## 3. Understanding Flow
- For a function: read its name, docstring, parameters, and return type first.
- For a class: read the `__init__` method to understand its dependencies.
- Follow the call chain from public APIs to internal implementations.

## 4. Documentation
- When explaining code to users, use simple language and avoid jargon.
- Include relevant file paths and line numbers in your explanation.
- Summarize **what** the code does and **why** it exists, not just **how**.

## 5. Code Navigation
- Prefer using grep to find references before reading full files.
- For large files, read the relevant section only (use line numbers).
- If you don't understand a dependency, explore it before making assumptions.
