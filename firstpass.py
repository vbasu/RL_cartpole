import math
import torch


class NextState(torch.nn.Module):
    def __init__(self):
        self.leftpush = torch.Parameter(torch.rand(4, 4), requires_grad=True)
        self.rightpush = torch.Parameter(torch.rand(4, 4), requires_grad=True)

    def forward(self, state, action):
        if action > 0:
            return torch.matmul(self.rightpush, state)
        else:
            return torch.matmul(self.leftpush, state)


class Value(torch.nn.Module):
    def __init__(self):
        self.coefficients = torch.Parameter(torch.rand(4), requires_grad=True)

    def forward(self, state):
        return torch.dot(self.coefficients, state)


class Policy(torch.nn.Module):
    def __init__(self):
        self.val = Value()
        self.next_state = NextState()
        self.possible_actions = [1, -1]

    def forward(self, state):
        value_func = lambda action: self.val(self.next_state(state, action))
        return max(self.possible_actions, key=value_func)


def acceptable(state):

    return abs(state[0]) < math.pi/12 and abs(state[2]) < 2.4


def actual_physics(state, action):

    g = 9.8
    L = 10
    dt = 0.01
    a = action
    theta, omega, x, v = state

    alpha = (g/2 * math.sin(theta) - a * math.cos(theta)) * 12 / (5*L)

    update_tensor = torch.tensor([omega, alpha, v, a])
    return state + update_tensor * dt


def state_loss(s1, s2):

    return torch.linalg.norm(s1-s2)


def gen_trajectory(starting_state, policy):

    trajectory = []
    state = starting_state
    while acceptable(state):
        action = policy(state)
        trajectory.append((state, action))
        state = actual_physics(state, action)
    trajectory.append((state, None))

    return trajectory


def train(starting_state=torch.tensor([0, 0, 0, 1])):

    policy = Policy()

    # Stuff for iteration
    for episode in range(10000):

        policy.freeze()  # Turns off gradient computation
        trajectory = gen_trajectory(starting_state, policy)
        policy.unfreeze()  # Questionable

        next = policy.next_state
        value = policy.val

        nxt_opt = torch.optim.Adam(next.parameters(), lr=0.1)
        val_opt = torch.optim.Adam(value.parameters(), lr=0.1)

        for i in range(len(trajectory) - 1)):
            state, action = trajectory[i]
            nextState = trajectory[i+1][0]

            state_loss(next(state, action), nextState).backward()
            nxt_opt.step()

            recurse_rhs = 1 + value(next(state, policy(state))).detach()
            recurse_lhs = value(state)
            val_loss = (recurse_lhs - recurse_rhs)**2
            val_loss.backward()
            val_opt.step()

    return policy
