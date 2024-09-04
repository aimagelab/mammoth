import os
import sys
import pytest
import logging


@pytest.fixture(autouse=True)
def setup_test_environ():
    # setting test environment variables
    os.environ['MAMMOTH_TEST'] = '1'
