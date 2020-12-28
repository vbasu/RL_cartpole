
import torch
import pytorch_lightning as pl

import environment as env


class NextState(pl.LightningModule):
    def __init__(self):
        self.leftpush = torch.Parameter(torch.rand(4, 4), requires_grad=True)
        self.rightpush = torch.Parameter(torch.rand(4, 4), requires_grad=True)

    def forward(self, state, action):
        if action > 0:
            return torch.matmul(self.rightpush, state)
        else:
            return torch.matmul(self.leftpush, state)


class Value(pl.LightningModule):
    def __init__(self):
        self.coefficients = torch.Parameter(torch.rand(4), requires_grad=True)

    def forward(self, state):
        return torch.dot(self.coefficients, state)


class Policy(pl.LightningModule):
    def __init__(self):
        self.val = Value()
        self.next_state = NextState()
        self.possible_actions = [1, -1]

    def forward(self, state):
        value_func = lambda action: self.val(self.next_state(state, action))
        return max(self.possible_actions, key=value_func)


def state_loss(s1, s2):

    return torch.linalg.norm(s1-s2)


def train():

    policy = Policy()
    world = env.Cartpole()

    # Stuff for iteration
    for episode in range(10000):

        policy.freeze()  # Turns off gradient computation
        trajectory = world.gen_trajectory(policy)
        policy.unfreeze()  # Turns on gradient computation

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
