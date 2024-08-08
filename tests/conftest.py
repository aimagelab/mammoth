import os
import pytest


@pytest.fixture(autouse=True)
def setup_test_environ():
    # setting test environment variables
    os.environ['MAMMOTH_TEST'] = '1'
