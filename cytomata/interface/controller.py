import time


class PID(object):
    """Proportional, integral, derivative controller
    with integral windup clipping"""

    def __init__(self, Kp, Ki, Kd, SP, windup_limit):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.SP = SP
        self.windup_limit = windup_limit
        self.ta = time.time()
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, PV):
        tb = time.time()
        dt = tb - self.ta
        self.ta = tb
        error = self.SP - PV
        integral = self.integral + error * dt
        self.integral = max(min(integral, windup_limit), -windup_limit)
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        output = Kp * error + Ki * self.integral + Kd * derivative
        return output
