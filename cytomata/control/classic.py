import time


class BB(object):
    """On-Off controller that turns the binary input ON when the output is below
    a certain threshold otherwise the input is turned OFF"""

    def __init__(self, th):
        self.th = th

    def step(self, PV):
        if PV < self.th:
            return True
        else:
            return False



class PID(object):
    """Proportional, integral, derivative controller
    with integral windup clipping"""

    def __init__(self, SP, Kp, Ki, Kd, windup_limit):
        self.SP = SP
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        # self.windup_limit = windup_limit
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, PV):
        error = self.SP - PV
        self.integral = self.integral + error
        # self.integral = max(min(self.integral, self.windup_limit), -self.windup_limit)
        derivative = (error - self.prev_error)
        self.prev_error = error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return output


class MPC(object):
    pass
