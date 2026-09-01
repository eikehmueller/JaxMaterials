# pytest configuration script
# Generated with the help of ChatGPT, reviewed by Eike Mueller
import pytest

OUTPUT_FILE = "testoutput.txt"


def pytest_sessionstart(session):
    # Truncate the file at the start of the test session
    open(OUTPUT_FILE, "w").close()


def pytest_runtest_logreport(report):
    if report.when != "call":
        return
    outcome = report.outcome.upper()
    if report.capstdout:
        with open(OUTPUT_FILE, "a") as f:
            f.write(f"\n==== {report.nodeid} : {outcome} ====\n")
            f.write(report.capstdout)
