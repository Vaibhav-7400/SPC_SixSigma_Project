import numpy as np


def nelson_rules(data, mean, sigma):
    data = np.asarray(data, dtype=float)
    n = len(data)
    results = {5: [], 6: [], 7: [], 8: []}

    # Rule 5: 6 consecutive points all increasing or all decreasing
    for i in range(5, n):
        diffs = np.diff(data[i - 5 : i + 1])
        if np.all(diffs > 0) or np.all(diffs < 0):
            results[5].append(i)

    # Rule 6: 14 consecutive points alternating up and down
    for i in range(13, n):
        diffs = np.diff(data[i - 13 : i + 1])
        signs = np.sign(diffs)
        if np.all(signs != 0) and np.all(signs[:-1] * signs[1:] < 0):
            results[6].append(i)

    # Rule 7: 15 consecutive points within 1 sigma (stratification)
    for i in range(14, n):
        window = data[i - 14 : i + 1]
        if np.all(np.abs(window - mean) < sigma):
            results[7].append(i)

    # Rule 8: 8 consecutive points beyond 1 sigma on both sides (mixture)
    for i in range(7, n):
        window = data[i - 7 : i + 1]
        if np.all(np.abs(window - mean) > sigma):
            results[8].append(i)

    return results
