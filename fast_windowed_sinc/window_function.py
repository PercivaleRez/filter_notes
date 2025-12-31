import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


def blackmanHarris(length: int):
    isEven = 1 - length % 2

    ω = 2 * np.pi / float(length - isEven)
    k = 2 * np.cos(ω)
    u1 = np.cos(-1 * ω)
    u2 = np.cos(-2 * ω)

    window = np.zeros(length)
    for i in range(length):
        u0 = k * u1 - u2
        u2 = u1
        u1 = u0
        window[i] = 0.21747 + u0 * (-0.45325 + u0 * (0.28256 + u0 * -0.04672))
    return window


def blackmanHarrisC(length: int, fraction: float):
    isEven = 1 - length % 2

    ω = 2 * np.pi / float(length - isEven)
    k = 2 * np.cos(ω)
    u1 = np.cos(ω * (-1 + fraction))
    u2 = np.cos(ω * (-2 + fraction))

    window = np.zeros(length)
    for i in range(length):
        u0 = k * u1 - u2
        u2 = u1
        u1 = u0
        window[i] = 0.21747 + u0 * (-0.45325 + u0 * (0.28256 + u0 * -0.04672))
    return window


def blackmanHarrisDirect(length: int, fraction: float):
    window = np.zeros(length)
    a_0 = 0.35875
    a_1 = 0.48829
    a_2 = 0.14128
    a_3 = 0.01168
    N = length - 1
    for i in range(N):
        x = (i + fraction) * np.pi / N
        val = a_0 - a_1 * np.cos(2 * x) + a_2 * np.cos(4 * x) - a_3 * np.cos(6 * x)
        window[i] = max(0.0, val)
    return window


def testBlackmanHarris():
    length = 64
    fraction = 0

    ref = signal.get_window("blackmanharris", length, fftbins=False)
    fast = blackmanHarris(length)
    fastC = blackmanHarrisC(length, fraction)
    direct = blackmanHarrisDirect(length, fraction)

    # plt.plot(ref, label="ref.")
    plt.plot(fast, label="fast")
    plt.plot(fastC, label="fastC")
    plt.plot(direct, label="direct")
    plt.axvline(
        (length - ((length + 1) % 2)) / 2,
        alpha=0.33,
        color="black",
        ls="--",
        label="center",
    )
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def triangle(length: int):
    tri_N: int = length + 2
    inv_N: float = 1 / tri_N
    window = np.zeros(length)
    for i in range(length):
        window[i] = 1 - abs((2 * (i + 1) - tri_N) * inv_N)
        # window[i] = 1.0 - abs((2 * i - length) * inv_N) # Simplified expression.
    return window


def triangleC(length: int, fraction: float):
    radius = (length + 2) / 2.0
    center = length / 2 - fraction
    window = np.zeros(length)
    for i in range(length):
        val = 1.0 - abs(i - center) / radius
        window[i] = max(0.0, val)
    return window


def testTriangle():
    length = 32

    ref = signal.get_window("triang", length)
    # fast = triangle(length)
    fast = triangleC(length, 1)

    plt.plot(ref, label="ref.")
    plt.plot(fast, label="fast")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def bartlett(length: int):
    tri_N: int = length
    inv_N: float = 1 / tri_N
    window = np.zeros(length)
    for i in range(length):
        window[i] = 1 - abs((2 * i - tri_N) * inv_N)
    return window


def triangle1(length: int):
    tri_N: int = length + 1
    inv_N: float = 1 / tri_N
    window = np.zeros(length)
    for i in range(length):
        window[i] = 1 - abs((2 * (i + 1) - tri_N) * inv_N)
    return window


def testBartlett():
    length = 31

    ref = signal.get_window("bartlett", length)
    fast = bartlett(length)

    plt.plot(ref, label="ref.")
    plt.plot(fast, label="fast")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def blackmanHarrisNarrow(length: int):
    ω = 2 * np.pi / float(length + 1)
    φ = np.pi / 2
    k = 2 * np.cos(ω)
    u1 = 1
    u2 = np.sin(φ - ω)

    window = np.zeros(length)
    for i in range(length):
        u0 = k * u1 - u2
        u2 = u1
        u1 = u0
        window[i] = 0.21747 + u0 * (-0.45325 + u0 * (0.28256 + u0 * -0.04672))
    return window


def blackmanHarrisTruncated(length: int, localTap: int):
    ω = 2 * np.pi / float(length + 1)
    φ = np.pi / 2 + ω * (length / 2 - localTap / 2)
    k = 2 * np.cos(ω)
    u1 = np.sin(φ)
    u2 = np.sin(φ - ω)

    window = np.zeros(localTap)
    for i in range(localTap):
        u0 = k * u1 - u2
        u2 = u1
        u1 = u0
        window[i] = 0.21747 + u0 * (-0.45325 + u0 * (0.28256 + u0 * -0.04672))
    return window


def testBlackmanHarrisPartial():
    length = 256

    ref = signal.get_window("blackmanharris", length + 2, fftbins=False)[1:-1]

    localTap = 64
    narrow = blackmanHarrisNarrow(localTap)
    trunc = blackmanHarrisTruncated(length, localTap)
    start = (length - localTap) / 2
    xp = np.arange(start, start + localTap)

    plt.figure(figsize=(6, 3))
    plt.plot(ref, label="Full", color="black")
    plt.plot(xp, narrow, label="Narrowing", color="blue", alpha=0.5)
    plt.plot(xp, trunc, label="Truncation", color="orange", lw=4, alpha=0.5)
    plt.axvline(start, alpha=0.33, color="black", ls="--")
    plt.axvline(start + localTap, alpha=0.33, color="black", ls="--")
    plt.title("Truncation of Blackman-Harris Window")
    plt.xlabel("Tap [sample]")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # testBlackmanHarris()
    # testTriangle()
    # testBartlett()
    testBlackmanHarrisPartial()
