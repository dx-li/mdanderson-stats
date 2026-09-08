"""Algorithm FL constants from RANDLIB; see retained third-party notices."""

import numpy as np
from numpy.typing import NDArray

from ._randlib_distributions import DistributionStream

A = np.array(
    [
        0.0,
        0.3917609e-1,
        0.7841241e-1,
        0.1177699,
        0.1573107,
        0.1970991,
        0.2372021,
        0.2776904,
        0.3186394,
        0.3601299,
        0.4022501,
        0.4450965,
        0.4887764,
        0.5334097,
        0.5791322,
        0.6260990,
        0.6744898,
        0.7245144,
        0.7764218,
        0.8305109,
        0.8871466,
        0.9467818,
        1.009990,
        1.077516,
        1.150349,
        1.229859,
        1.318011,
        1.417797,
        1.534121,
        1.675940,
        1.862732,
        2.153875,
    ],
    dtype=np.float32,
)
D = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2636843,
        0.2425085,
        0.2255674,
        0.2116342,
        0.1999243,
        0.1899108,
        0.1812252,
        0.1736014,
        0.1668419,
        0.1607967,
        0.1553497,
        0.1504094,
        0.1459026,
        0.1417700,
        0.1379632,
        0.1344418,
        0.1311722,
        0.1281260,
        0.1252791,
        0.1226109,
        0.1201036,
        0.1177417,
        0.1155119,
        0.1134023,
        0.1114027,
        0.1095039,
    ],
    dtype=np.float32,
)
H = np.array(
    [
        0.3920617e-1,
        0.3932705e-1,
        0.3950999e-1,
        0.3975703e-1,
        0.4007093e-1,
        0.4045533e-1,
        0.4091481e-1,
        0.4145507e-1,
        0.4208311e-1,
        0.4280748e-1,
        0.4363863e-1,
        0.4458932e-1,
        0.4567523e-1,
        0.4691571e-1,
        0.4833487e-1,
        0.4996298e-1,
        0.5183859e-1,
        0.5401138e-1,
        0.5654656e-1,
        0.5953130e-1,
        0.6308489e-1,
        0.6737503e-1,
        0.7264544e-1,
        0.7926471e-1,
        0.8781922e-1,
        0.9930398e-1,
        0.1155599,
        0.1404344,
        0.1836142,
        0.2790016,
        0.7010474,
    ],
    dtype=np.float32,
)
T = np.array(
    [
        0.7673828e-3,
        0.2306870e-2,
        0.3860618e-2,
        0.5438454e-2,
        0.7050699e-2,
        0.8708396e-2,
        0.1042357e-1,
        0.1220953e-1,
        0.1408125e-1,
        0.1605579e-1,
        0.1815290e-1,
        0.2039573e-1,
        0.2281177e-1,
        0.2543407e-1,
        0.2830296e-1,
        0.3146822e-1,
        0.3499233e-1,
        0.3895483e-1,
        0.4345878e-1,
        0.4864035e-1,
        0.5468334e-1,
        0.6184222e-1,
        0.7047983e-1,
        0.8113195e-1,
        0.9462444e-1,
        0.1123001,
        0.1364980,
        0.1716886,
        0.2276241,
        0.3304980,
        0.5847031,
    ],
    dtype=np.float32,
)
C_A = np.array(
    [
        0.0,
        3.917609e-2,
        7.841241e-2,
        0.11777,
        0.1573107,
        0.1970991,
        0.2372021,
        0.2776904,
        0.3186394,
        0.36013,
        0.4022501,
        0.4450965,
        0.4887764,
        0.5334097,
        0.5791322,
        0.626099,
        0.6744898,
        0.7245144,
        0.7764218,
        0.8305109,
        0.8871466,
        0.9467818,
        1.00999,
        1.077516,
        1.150349,
        1.229859,
        1.318011,
        1.417797,
        1.534121,
        1.67594,
        1.862732,
        2.153875,
    ],
    dtype=np.float32,
)
C_H = np.array(
    [
        3.920617e-2,
        3.932705e-2,
        3.951e-2,
        3.975703e-2,
        4.007093e-2,
        4.045533e-2,
        4.091481e-2,
        4.145507e-2,
        4.208311e-2,
        4.280748e-2,
        4.363863e-2,
        4.458932e-2,
        4.567523e-2,
        4.691571e-2,
        4.833487e-2,
        4.996298e-2,
        5.183859e-2,
        5.401138e-2,
        5.654656e-2,
        5.95313e-2,
        6.308489e-2,
        6.737503e-2,
        7.264544e-2,
        7.926471e-2,
        8.781922e-2,
        9.930398e-2,
        0.11556,
        0.1404344,
        0.1836142,
        0.2790016,
        0.7010474,
    ],
    dtype=np.float32,
)
C_T = np.array(
    [
        7.673828e-4,
        2.30687e-3,
        3.860618e-3,
        5.438454e-3,
        7.0507e-3,
        8.708396e-3,
        1.042357e-2,
        1.220953e-2,
        1.408125e-2,
        1.605579e-2,
        1.81529e-2,
        2.039573e-2,
        2.281177e-2,
        2.543407e-2,
        2.830296e-2,
        3.146822e-2,
        3.499233e-2,
        3.895483e-2,
        4.345878e-2,
        4.864035e-2,
        5.468334e-2,
        6.184222e-2,
        7.047983e-2,
        8.113195e-2,
        9.462444e-2,
        0.1123001,
        0.136498,
        0.1716886,
        0.2276241,
        0.330498,
        0.5847031,
    ],
    dtype=np.float32,
)

for table in (A, D, H, T, C_A, C_H, C_T):
    table.flags.writeable = False


def _threshold(w: np.float32, aa: np.float32, c_source: bool) -> np.float32:
    if c_source:
        return np.float32((0.5 * float(w) + float(aa)) * float(w))
    return (np.float32(0.5) * w + aa) * w


def standard_normal(stream: DistributionStream) -> np.float32:
    """Original FL center/tail rejection algorithm with local sampling state."""
    c_source = stream.source == "c"
    a, h, t = (C_A, C_H, C_T) if c_source else (A, H, T)
    u = stream.uniform()
    sign = np.float32(u > 0.5)
    u = (u + (u - sign)) if c_source else (u + u - sign)
    u = np.float32(32) * u
    i = min(int(u), 31)
    if i:
        ustar = u - np.float32(i)
        aa = a[i - 1]
        while True:
            if ustar > t[i - 1]:
                w = (ustar - t[i - 1]) * h[i - 1]
                break
            u = stream.uniform()
            w = u * (a[i] - aa)
            tt = _threshold(w, aa, c_source)
            while ustar <= tt:
                u = stream.uniform()
                if ustar < u:
                    ustar = stream.uniform()
                    break
                tt = u
                ustar = stream.uniform()
            else:
                break
    else:
        i, aa = 5, a[31]
        while True:
            u = u + u
            if u >= 1:
                break
            aa = aa + D[i]
            i += 1
            if i >= len(D):
                raise ArithmeticError("legacy normal exceeds its source table; state unchanged")
        u = u - np.float32(1)
        while True:
            w = u * D[i]
            tt = _threshold(w, aa, c_source)
            while True:
                ustar = stream.uniform()
                if ustar > tt:
                    y = aa + w
                    return -y if sign else y
                u = stream.uniform()
                if ustar < u:
                    u = stream.uniform()
                    break
                tt = u
    y = aa + w
    return -y if sign else y


def legacy_normal(
    stream: DistributionStream, size: int, mean: np.float32, sd: np.float32
) -> NDArray[np.float64]:
    result = np.empty(size, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for i in range(size):
            result[i] = sd * standard_normal(stream) + mean
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("normal samples overflow float32; state unchanged")
    return result
