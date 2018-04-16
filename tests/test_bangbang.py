import pytest
import numpy as np
import matplotlib.pyplot as plt
from cytomata.control import BangBang
from cytomata.simulate import Simple


def test_bb_maintains_setpoint():
    """Test that the BB controller can drive and maintain the process variable
    within an error interval around the setpoint."""
    SP = 500.0
    sim0 = Simple()
    bb = BangBang(SP * 0.98)
    x = []
    y = []
    timerange = np.arange(0, 200, 1)

    for t in timerange:
        xi = sim0.bl
        yi = sim0.step()
        sim0.bl = bb.step(yi) * SP * 0.25
        x.append(xi)
        y.append(yi)

    plt.axhline(y=SP, color='r')
    plt.plot(timerange, x, label='input')
    plt.plot(timerange, y, label='output')
    plt.legend()
    plt.show()

    assert y[-1] > SP - 0.1 * SP and y[-1] < SP + 0.1 * SP


def test_pid_accomodates_perturbation():
    """Test that the BB controller can return system to setpoint after it has
    been perturbed."""
    SP = 500.0
    TH_scale = 0.98
    MV_scale = 0.2
    sim0 = Simple()
    sim0.mc = SP
    bb = BangBang(SP * TH_scale)
    x = []
    y = []
    timerange = np.arange(0, 200, 1)

    for t in timerange:
        if t % 20 == 0 and t > timerange[20] and t < timerange[-20]:
            noise = np.random.normal(0.0, 100.0)
            sim0.mc += noise
        xi = sim0.bl
        yi = sim0.step()
        sim0.bl = bb.step(yi) * SP * MV_scale
        x.append(xi)
        y.append(yi)

    plt.axhline(y=SP, color='r')
    plt.plot(timerange, x, label='input')
    plt.plot(timerange, y, label='output')
    plt.legend()
    plt.show()

    assert y[-1] > SP - 0.1 * SP and y[-1] < SP + 0.1 * SP


def test_pid_tracks_moving_setpoint():
    """Test that the BB controller can maintain the process variable within
    the error interval around a changing setpoint."""
    SP = 500.0
    TH_scale = 0.98
    MV_scale = 0.2
    sim0 = Simple()
    sim0.mc = SP
    bb = BangBang(SP * TH_scale)
    x = []
    y = []
    th = []
    timerange = np.arange(0, 200, 1)

    for t in timerange:
        if t == 20:
            bb.th += 20
        if t == 60:
            bb.th -= 40
        if t == 100:
            bb.th += 60
        if t == 140:
            bb.th -= 120
        xi = sim0.bl
        yi = sim0.step()
        sim0.bl = bb.step(yi) * SP * MV_scale
        x.append(xi)
        y.append(yi)
        th.append(bb.th)

    plt.plot(timerange, x, label='input')
    plt.plot(timerange, y, label='output')
    plt.plot(timerange, th, label='setpoint')
    plt.legend()
    plt.show()

    assert y[-1] > bb.th - 0.1 * bb.th and y[-1] < bb.th + 0.1 * bb.th
