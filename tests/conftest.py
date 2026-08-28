"""Phase 0 scaffolding — remove once real tests exist.

pytest exits 5 ("no tests collected") on an empty suite, which would fail CI
before Phase 1 lands the first test file. Translate that one status into 0 so
Phase 0's exit criterion ("pytest runs with zero tests, exit 0") holds.
Inert as soon as a single test is collected.
"""

EXIT_NO_TESTS_COLLECTED = 5


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == EXIT_NO_TESTS_COLLECTED:
        session.exitstatus = 0
