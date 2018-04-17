import pytest
import numpy as np
import matplotlib.pyplot as plt
from cytomata.control import BangBang
from cytomata.simulate import Simple, OptoLexA


@pytest.mark.parametrize('model, bb, mv_scale', [
    (Simple(), BangBang(500.0 * 0.98), 0.25),
    (OptoLexA(), BangBang(500.0 * 0.98), 0.1),
])
def test_bb_maintains_setpoint(model, bb, mv_scale):
    """Test that the BB controller can drive and maintain the process variable
    within an error interval around the setpoint."""
    SP = 500.0
    x = []
    y = []
    timerange = np.arange(0, 200, 1)

    for t in timerange:
        xi = model.bl
        yi = model.step()[0]
        model.bl = bb.step(yi) * SP * mv_scale
        x.append(xi)
        y.append(yi)

    plt.axhline(y=SP, color='r')
    plt.plot(timerange, x, label='input')
    plt.plot(timerange, y, label='output')
    plt.legend()
    plt.show()

    assert y[-1] > SP - 0.1 * SP and y[-1] < SP + 0.1 * SP


@pytest.mark.parametrize('model, bb, mv_scale', [
    (Simple(), BangBang(500.0 * 0.98), 0.25),
    (OptoLexA(), BangBang(500.0 * 0.9), 0.1),
])
def test_bb_accomodates_perturbation(model, bb, mv_scale):
    """Test that the BB controller can return system to setpoint after it has
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
        model.bl = bb.step(yi) * SP * mv_scale
        x.append(xi)
        y.append(yi)

    plt.axhline(y=SP, color='r')
    plt.plot(timerange, x, label='input')
    plt.plot(timerange, y, label='output')
    plt.legend()
    plt.show()

    assert y[-1] > SP - 0.1 * SP and y[-1] < SP + 0.1 * SP


@pytest.mark.parametrize('model, bb, mv_scale', [
    (Simple(), BangBang(500.0 * 0.98), 0.25),
    (OptoLexA(), BangBang(500.0 * 0.9), 0.05),
])
def test_bb_tracks_moving_setpoint(model, bb, mv_scale):
    """Test that the BB controller can maintain the process variable within
    the error interval around a changing setpoint."""
    SP = 500.0
    model.mc = SP
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
        xi = model.bl
        yi = model.step()[0]
        model.bl = bb.step(yi) * SP * mv_scale
        x.append(xi)
        y.append(yi)
        th.append(bb.th)

    plt.plot(timerange, x, label='input')
    plt.plot(timerange, y, label='output')
    plt.plot(timerange, th, label='setpoint')
    plt.legend()
    plt.show()

    assert y[-1] > bb.th - 0.1 * bb.th and y[-1] < bb.th + 0.1 * bb.th
