
import torch
from math import pi, cos, sin


class Environment(object):
    def __init__(self):
        # Ensure that subclass has appropriately defined state property
        try:
            cache_state = self.state
            self.state = cache_state
        except:
            print(f"State property not appropriately defined for {self}")
            raise

    def with_custom_state(func):
        def wrapper(self, *args, state=None, **kwargs):
            if state is not None:
                cache_state = self.state
                self.state = state
            result = func(self, *args, **kwargs)
            if state is not None:
                self.state = cache_state
            return result
        return wrapper

    @with_custom_state
    def game_over(self):
        raise NotImplementedError

    @with_custom_state
    def step(self):
        raise NotImplementedError

    @with_custom_state
    def gen_trajectory(self, policy, max_turns=1000):

        trajectory = []
        for i in range(max_turns):
            if self.game_over(): break
            trajectory.append((self.state, policy(self.state)))
            self.step(trajectory[-1][1])
        trajectory.append((self.state, None))

        return trajectory


class Cartpole(Environment):
    def __init__(self, theta=0, omega=0, x=0, v=1, g=9.8, L=10):
        self.theta = theta  # angle of deflection of pole from vertical
        self.omega = omega  # angular velocity of pole around pivot
        self.x = x  # horizontal position of cart
        self.v = v  # velocity of cart
        self.g = g  # gravitational acceleration
        self.L = L  # length of pole

    @property
    def state(self):
        return torch.tensor([theta, omega, x, v])

    @state.setter
    def state(self, state_tensor):
        self.theta, self.omega, self.x, self.v = state_tensor.tolist()

    @with_custom_state
    def game_over(self):
        return abs(self.theta) >= pi/12 or abs(self.x) >= 2.4

    @with_custom_state
    def step(self, action, dt=0.01):
        a = action  # acceleration applied to cart
        tau = (self.g * sin(self.theta) - self.a * cos(self.theta)) * self.L / 2
        rotI = L**2 / 3
        alpha = tau / rotI

        update_tensor = torch.tensor([omega, alpha, v, a])
        self.state = self.state + update_tensor * dt
