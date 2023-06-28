import numpy as np

data = '011291831741091812173109418711012918378'
counts = [0 for _ in range(10)]
for d in data:
    counts[int(d)] += 1
print(counts)

num_total = len(data)
probs = [c / float(num_total) for c in counts]
print(probs)

entropy = 0
eps = 1e-6
for p in probs:
    entropy += p * (-1) * np.log2(p + eps)
print('entropy = ', entropy)
