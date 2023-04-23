import pytest
import os

from src.main import IPSRecording
from src.tests.utils.ips_utils import compute_path


@pytest.fixture
def ips():
    PB_FILE = compute_path(r'../recordings_pb/10732.pb')
    return IPSRecording(os.path.join(os.getcwd(), PB_FILE))

@pytest.fixture
def expected_magnetics_columns():
    return ['t', 'mx', 'my', 'mz', 'accuracy', 'x', 'y']

@pytest.fixture
def expected_positions_columns():
    return ['t', 'x', 'y', 'floor', 'type', 'accuracy']