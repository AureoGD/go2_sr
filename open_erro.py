import numpy as np
q     = np.array([0.01884159, 0.00524193, 0.96941993, 0.24462732])
q_ref = np.array([0.0037, 0.0053, 0.9701, 0.2429])

def qmul(a, b):  # xyzw
    x1,y1,z1,w1 = a; x2,y2,z2,w2 = b
    return np.array([w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2,
                     w1*z2+x1*y2-y1*x2+z1*w2,
                     w1*w2-x1*x2-y1*y2-z1*z2])

def qconj(a):
    return np.array([-a[0], -a[1], -a[2], a[3]])

eps = qmul(qconj(q_ref), q)
print(eps)   # should be ~[0.015, ~0, ~0, ~1]