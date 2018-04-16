import pytest
import numpy as np
import matplotlib.pyplot as plt
from cytomata.control import PID
from cytomata.simulate import Simple


def test_pid_maintains_setpoint():
    """Test that the PID controller can drive and maintain the process variable
    within an error interval around the setpoint."""
    sim0 = Simple()
    SP = 500.0
    pid = PID(Kp=1.1, Ki=0.1, Kd=0.1, SP=SP, windup_limit=20.0)
    x = []
    y = []
    timerange = np.arange(0, 100, 1)

    for t in timerange:
        xi = sim0.bl
        yi = sim0.step()
        sim0.bl = pid.step(yi)
        x.append(xi)
        y.append(yi)

    assert y[-1] > SP - 0.05 * SP and y[-1] < SP + 0.05 * SP

    plt.axhline(y=SP, color='r')
    plt.plot(timerange, x, label='input')
    plt.plot(timerange, y, label='output')
    plt.legend()
    plt.show()


def test_pid_accomodates_perturbation():
    """Test that the PID controller can return system to setpoint after it has
    been perturbed."""
    SP = 500.0
    sim0 = Simple()
    sim0.mc = SP
    pid = PID(Kp=1.0, Ki=0.5, Kd=0, SP=SP, windup_limit=20.0)
    x = []
    y = []
    timerange = np.arange(0, 200, 1)

    for t in timerange:
        if t % 20 == 0 and t > timerange[20] and t < timerange[-20]:
            noise = np.random.normal(0.0, 100.0)
            sim0.mc += noise
        xi = sim0.bl
        yi = sim0.step()
        sim0.bl = pid.step(yi)
        x.append(xi)
        y.append(yi)

    assert y[-1] > SP - 0.05 * SP and y[-1] < SP + 0.05 * SP

    plt.axhline(y=SP, color='r')
    plt.plot(timerange, x, label='input')
    plt.plot(timerange, y, label='output')
    plt.legend()
    plt.show()


def test_pid_tracks_moving_setpoint():
    """Test that the PID controller can maintain the process variable within
    the error interval around a changing setpoint."""
    SP = 500.0
    sim0 = Simple()
    sim0.mc = SP
    pid = PID(Kp=1.0, Ki=0.1, Kd=0, SP=SP, windup_limit=20.0)
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
        xi = sim0.bl
        yi = sim0.step()
        sim0.bl = pid.step(yi)
        x.append(xi)
        y.append(yi)
        sp.append(pid.SP)

    plt.plot(timerange, x, label='input')
    plt.plot(timerange, y, label='output')
    plt.plot(timerange, sp, label='setpoint')
    plt.legend()
    plt.show()

    assert y[-1] > pid.SP - 0.05 * pid.SP and y[-1] < pid.SP + 0.05 * pid.SP
