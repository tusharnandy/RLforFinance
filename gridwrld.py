import numpy as np

# Functions with return new positions based on action
# and current location, with boundary constraints
def up(y,x):
    if y<=0:
        return y,x
    else:
        return (y-1),x

def down(y,x):
    if y >= 3:
        return y,x
    else:
        return (y+1),x

def right(y,x):
    if x >= 3:
        return y,x
    else:
        return y, (x+1)

def left(y,x):
    if x <= 0:
        return y,x
    else:
        return y, (x-1)

# Policy evaluation:
# In this case, pi(a|s) = 0.25 in each state
 
def update_p1(v):
    """
    Args:
        v: 4x4 grid of with each value denoting the value function
           of that spot.

    returns:
        W: updated policy grid
    """
    w = np.zeros((4,4))
    for y in range(4):
        for x in range(4):
            if (y == x) and ((y+x) % 6) == 0:   # stop in terminal states
                continue
            score = -1
            score += 0.25*(v[up(y,x)] + v[down(y,x)])
            score += 0.25*(v[right(y,x)] + v[left(y,x)])
            w[y,x] = np.round(score, 2)
    return w


v = np.zeros((4,4))
for count in range(70):
