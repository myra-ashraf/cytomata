import pytest
import numpy as np
import matplotlib.pyplot as plt
from cytomata.control import PID
from cytomata.simulate import Simple, OptoLexA


@pytest.mark.parametrize('model, pid', [
    (Simple(), PID(SP=500.0, Kp=1.1, Ki=0.1, Kd=0.1, windup_limit=20.0)),
    (OptoLexA(), PID(SP=500.0, Kp=2.0, Ki=0.2, Kd=2.0, windup_limit=20.0)),
])
def test_pid_maintains_setpoint(model, pid):
    """Test that the PID controller can drive and maintain the process variable
    within an error interval around the setpoint."""
    SP = 500.0
    x = []
    y = []
    timerange = np.arange(0, 100, 1)

    for t in timerange:
        xi = model.bl
        yi = model.step()[0]
        model.bl = pid.step(yi)
        x.append(xi)
        y.append(yi)

    plt.axhline(y=SP, color='r')
    plt.plot(timerange, x, label='input')
    plt.plot(timerange, y, label='output')
    plt.legend()
    plt.show()

    assert y[-1] > SP - 0.05 * SP and y[-1] < SP + 0.05 * SP


@pytest.mark.parametrize('model, pid', [
    (Simple(), PID(SP=500.0, Kp=1.1, Ki=0.1, Kd=0.1, windup_limit=20.0)),
    (OptoLexA(), PID(SP=500.0, Kp=2.0, Ki=0.2, Kd=2.0, windup_limit=20.0)),
])
def test_pid_accomodates_perturbation(model, pid):
    """Test that the PID controller can return system to setpoint after it has
    been perturbed."""
    SP = 500.0
    model.mc = SP
    x = []
    y = []
    timerange = np.arange(0, 200, 1)

    for t in timerange:
        if t % 20 == 0 and t > timerange[20] and t < timerange[-20]:
            noise = np.random.normal(0.0, 100.0)
            model.mc += noise
        xi = model.bl
        yi = model.step()[0]
        model.bl = pid.step(yi)
        x.append(xi)
        y.append(yi)

    assert y[-1] > SP - 0.05 * SP and y[-1] < SP + 0.05 * SP

    plt.axhline(y=SP, color='r')
    plt.plot(timerange, x, label='input')
    plt.plot(timerange, y, label='output')
    plt.legend()
    plt.show()


@pytest.mark.parametrize('model, pid', [
    (Simple(), PID(SP=500.0, Kp=1.1, Ki=0.1, Kd=0.1, windup_limit=20.0)),
    (OptoLexA(), PID(SP=500.0, Kp=0.5, Ki=0.1, Kd=2.0, windup_limit=20.0)),
])
def test_pid_tracks_moving_setpoint(model, pid):
    """Test that the PID controller can maintain the process variable within
    the error interval around a changing setpoint."""
    SP = 500.0
    model.mc = SP
    x = []
    y = []
    sp = []
    timerange = np.arange(0, 200, 1)

    for t in timerange:
        if t == 20:
            pid.SP += 20
        if t == 60:
            pid.SP -= 40
        if t == 100:
            pid.SP += 60
        if t == 140:
            pid.SP -= 120
        xi = model.bl
        yi = model.step()[0]
        model.bl = pid.step(yi)
        x.append(xi)
        y.append(yi)
        sp.append(pid.SP)

    plt.plot(timerange, x, label='input')
    plt.plot(timerange, y, label='output')
    plt.plot(timerange, sp, label='setpoint')
    plt.legend()
    plt.show()

    assert y[-1] > pid.SP - 0.05 * pid.SP and y[-1] < pid.SP + 0.05 * pid.SP
