import pytest
from cytomata.control import PID


def test_pid_maintains_setpoint(pid):
    """Test that the PID controller can drive and maintain the process variable
    within an error interval around the setpoint."""
    pass


def test_pid_accomodates_perturbation(pid):
    """Test that the PID controller can return system to setpoint after it has
    been perturbed."""
    pass


def test_pid_tracks_moving_setpoint(pid):
    """Test that the PID controller can maintain the process variable within
    the error interval around a changing setpoint."""
    pass
