# RL_cartpole
For OpenAI's cartpole problem


Cart has some mass M and we push with our force F

Pole has mass m and is free to rotate.



5/12 m*L^2 * alpha = - L * m * a * cos(theta) + L/2 * m * g * sin(theta)


alpha = [ g/2 sin(theta) - a cos(theta) ] * 12 / (5L)





-----------

Time step is fixed =P

Theta, Omega, x, and v comprise the state

next_state = f(state, action)
value(state)


policy p(state) = action that you would take

Let p(state) be a deterministic function that gives us 1 or -1,
or a thing that gives us two probabilities and we sample

value(state) = immediate_value(p(state)) + r * value(next_state) for r <= 1

value(state) = max over all actions of immediate_value(action) + r * value(next_state) for r <= 1 (1)



Way we train:

Generate history with policy
For each transition (s1 -> s2 with action A) in the history, we train

- f(s1, A) to be close to s2
- value(s1) to be close to RHS of (1) using one of two options
* state 1, action A (this used to be the arg max), state 2
* state 1, action A' based on a new arg max for the current value function, next_state = f(state 1, A')









F = (M+m)*a
