import pytest
from cytomata.interface import Microscope
from cytomata.control import PID

@pytest.fixture
def pid():
    """Set up PID controller object for tests."""
    pid = PID(Kp=100.0, Ki=10.0, Kd=1.0, SP=500.0, windup_limit=20.0)
    yield pid


@pytest.fixture
def mic():
    """Set up microscope MM object for tests."""
    mic = Microscope()
    yield mic
