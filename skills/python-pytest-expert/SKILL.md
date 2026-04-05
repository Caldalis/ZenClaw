---
name: python-pytest-expert
description: Use this skill when you need to write, run, or debug pytest unit tests in Python.
---

# Pytest Expert Skill

You are now operating with the Pytest Expert skill. Follow these strict workflows:

## 1. Test File Naming
- All test files **MUST** start with `test_` and be placed in the `tests/` directory.
- Test class names must start with `Test`, e.g., `TestCalculator`.

## 2. Execution Command
- Always use `python -m pytest -v -s` to run tests. Do not use plain `pytest`.
- For specific files: `python -m pytest tests/test_foo.py -v`

## 3. Mocking Rules
- Prefer using `unittest.mock.patch` over other mocking libraries.
- If a test fails with a networking error, you **MUST** mock the external HTTP request immediately. Do not attempt to fix the network.

## 4. Test Structure
- Use `pytest.fixture` for shared setup/teardown.
- Keep tests focused: one assertion per test when possible.
- Use `pytest.mark.parametrize` for data-driven tests.

## 5. Assertions
- Always import assertions: `from pytest import assert_equal, assert_true, assertRaises`
- Never use bare `assert` without a descriptive message for complex checks.
