"""
Pytest configuration for integration tests.

This file contains pytest hooks and fixtures that are shared across
the integration test suite.
"""

import pytest


def pytest_addoption(parser):
    """
    Add custom command line options for integration tests.
    
    This hook is called once at the start of a test run.
    """
    parser.addoption(
        "--protocol",
        action="store",
        default="test_baseline_proT_ishigami_cat",
        help="Protocol experiment to test for sweep compatibility (default: test_baseline_proT_ishigami_cat)"
    )
