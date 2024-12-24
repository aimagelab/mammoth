import os
import pytest


@pytest.fixture(autouse=True)
def setup_test_environ():
    # setting test environment variables
    os.environ['MAMMOTH_TEST'] = '1'


@pytest.fixture(autouse=True)
def force_device(request):
    ext_device = request.config.getoption("--force_device")
    if ext_device is not None:
        os.environ['MAMMOTH_DEVICE'] = str(ext_device)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(ext_device)


def pytest_addoption(parser):
    parser.addoption("--include_dataset_reload", default=False, action="store_true", help="Include dataset reload tests? NOTE: It will take a long time to run and will delete and reload the dataset.")
    parser.addoption("--force_device", default=None, type=str, help="Force device to use for testing.")


def pytest_collection_modifyitems(config, items):
    reload_datasets = config.getoption("--include_dataset_reload")
    if reload_datasets:
        print("\nIncluding dataset reload tests.\n")
    else:
        print("\nExcluding dataset reload tests.\n")

    skip_listed = pytest.mark.skip()
    for item in items:
        if not reload_datasets:
            if item.name == 'test_datasets_with_download':
                item.add_marker(skip_listed)
        else:
            if item.name == 'test_datasets_without_download':
                item.add_marker(skip_listed)
