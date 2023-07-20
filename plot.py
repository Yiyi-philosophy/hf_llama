import numpy as np
import matplotlib.pyplot as plt

b =np.array([
    [[1,2,3,4,5],
     [6,7,8,9,10]],
    [[11,12,13,14,15],
     [16,17,18,19,110]],
    [[21,22,23,24,25],
     [36,47,58,39,210]],
    ])
print(b)

plt.xlabel("seq index")
plt.ylabel("PPL")
plt.plot(b[2, 0, :], b[2, 1, :], label="NTK")
plt.plot(b[1, 0, :], b[1, 1, :], label="PI")
plt.plot(b[0, 0, :], b[0, 1, :], label="Standard")
plt.legend()


plt.savefig('./test_ntk,jpg')

