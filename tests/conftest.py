import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "mock: tests driven ONLY by the MockNavPolicy (no real model evidence)"
    )
