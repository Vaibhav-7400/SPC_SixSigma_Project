import numpy as np


def weco_rules(data, mean, sigma):
    data = np.asarray(data, dtype=float)
    n = len(data)
    results = {1: [], 2: [], 3: [], 4: []}

    # Rule 1: 1 point beyond 3 sigma
    for i in range(n):
        if abs(data[i] - mean) > 3 * sigma:
            results[1].append(i)

    # Rule 2: 2 of 3 consecutive points beyond 2 sigma on the same side
    for i in range(2, n):
        window = data[i - 2 : i + 1]
        if np.sum(window > mean + 2 * sigma) >= 2 or np.sum(window < mean - 2 * sigma) >= 2:
            results[2].append(i)

    # Rule 3: 4 of 5 consecutive points beyond 1 sigma on the same side
    for i in range(4, n):
        window = data[i - 4 : i + 1]
        if np.sum(window > mean + sigma) >= 4 or np.sum(window < mean - sigma) >= 4:
            results[3].append(i)

    # Rule 4: 8 consecutive points on the same side of the centerline
    for i in range(7, n):
        window = data[i - 7 : i + 1]
        if np.all(window > mean) or np.all(window < mean):
            results[4].append(i)

    return results
