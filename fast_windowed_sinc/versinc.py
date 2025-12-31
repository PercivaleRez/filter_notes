"""
TODO: Test fast version of versinc.
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import tqdm
import soundfile
from pathlib import Path


class Delay:
    maxTap = 256
    maxCutoffOctave: int = int(np.round(np.log2(maxTap)))
    recursiveOscFractionMrgin = 1024 * np.finfo(np.float64).eps

    def __init__(self, maxTimeSample: int):
        self.prevTime: float = 0
        self.wptr: int = 0
        self.buf: np.ndarray = np.zeros(
            max(self.maxTap, maxTimeSample + self.maxTap // 2 + 1)
        )

        self.minTimeSample: int = self.maxTap // 2 - (self.maxTap + 1) % 2
        self.maxTimeSample: int = maxTimeSample

    def processVersinc(self, input: float, timeInSample: float):
        size: int = len(self.buf)

        # Write to buffer.
        self.wptr += 1
        if self.wptr >= size:
            self.wptr = 0
        self.buf[self.wptr] = input

        # Read from buffer.
        localTap = np.clip(2 * int(timeInSample), 2, self.maxTap)
        halfTap: int = localTap // 2
        clamped: float = np.clip(timeInSample, halfTap - 1, self.maxTimeSample)
        timeInt: int = int(clamped)

        timeDiff: float = abs(self.prevTime - clamped + 1)
        self.prevTime = clamped
        cutoff: float = 0.5 if timeDiff <= 1 else 2.0 ** (-timeDiff)
        if timeInSample <= 0:
            return 0
        fir = lowpassWindowedReversed(localTap, cutoff, clamped - timeInt)

        rptr: int = self.wptr - timeInt - halfTap
        if rptr < 0:
            rptr += size

        sig = 0
        for idx in range(localTap):
            sig += fir[idx] * self.buf[rptr]
            rptr += 1
            if rptr >= size:
                rptr = 0
        return sig

    def processVersincBlackmanHarris(self, input: float, timeInSample: float):
        size: int = len(self.buf)

        # Write to buffer.
        self.wptr += 1
        if self.wptr >= size:
            self.wptr = 0
        self.buf[self.wptr] = input

        # Read from buffer.
        localTap = np.clip(2 * int(timeInSample), 2, self.maxTap)
        halfTap: int = localTap // 2
        clamped: float = np.clip(timeInSample, halfTap - 1, self.maxTimeSample)
        timeInt: int = int(clamped)

        timeDiff: float = abs(self.prevTime - clamped + 1)
        self.prevTime = clamped
        cutoff: float = 0.5 if timeDiff <= 1 else 2.0 ** (-timeDiff)
        if timeInSample <= 0:
            return 0

        rptr: int = self.wptr - timeInt - halfTap
        if rptr < 0:
            rptr += size

        # Setup recursive sine oscillator.
        fraction = clamped - timeInt
        mid = fraction - halfTap

        o1_omega = np.pi * cutoff
        o1_phi = mid * o1_omega
        o1_u1 = np.sin(o1_phi - o1_omega)
        o1_u2 = np.sin(o1_phi - 2 * o1_omega)
        o1_k = 2 * np.cos(o1_omega)

        # o2_omega = 2 * np.pi / localTap
        # o2_u1 = np.cos(o2_omega * (-1.0 + fraction))
        # o2_u2 = np.cos(o2_omega * (-2.0 + fraction))
        # o2_k = 2 * np.cos(o2_omega)

        sig = 0
        for idx in range(localTap):
            o1_u0 = o1_k * o1_u1 - o1_u2
            o1_u2 = o1_u1
            o1_u1 = o1_u0

            # o2_u0 = o2_k * o2_u1 - o2_u2
            # o2_u2 = o2_u1
            # o2_u1 = o2_u0

            # window = 0.21747 + o2_u0 * (-0.45325 + o2_u0 * (0.28256 + o2_u0 * -0.04672))

            kernel = 0
            x = idx + mid
            theta = np.pi * cutoff * x
            if abs(theta) < 0.05 * np.pi:
                t2 = theta * theta
                poly = -2.0 / 467775.0
                poly = poly * t2 + 2.0 / 14175.0
                poly = poly * t2 - 1.0 / 315.0
                poly = poly * t2 + 2.0 / 45.0
                poly = poly * t2 - 1.0 / 3.0
                poly = poly * t2 + 1.0
                kernel = (2 * cutoff * theta) * poly
            else:
                kernel = 2 * o1_u0 * o1_u0 / (np.pi * x)

            sig += kernel * self.buf[rptr]
            rptr += 1
            if rptr >= size:
                rptr = 0
        return sig


def plotResponse(firList, cutoffList, nameList, worN=8192, fs=48000, title=""):
    fig, ax = plt.subplots(4, 1)
    if len(title) >= 1:
        fig.suptitle(title)

    # worN = np.hstack([[0], np.geomspace(0.1, fs / 2)])

    gdMedians = []
    cmap = plt.get_cmap("plasma")
    for idx, fir in enumerate(firList):
        freq, resp = signal.freqz(fir, 1, worN=worN, fs=fs)
        freq, delay = signal.group_delay((fir, 1), w=worN, fs=fs)
        gain = 20 * np.log10(np.abs(resp))
        phase = np.unwrap(np.angle(resp))

        cut = max(1, int(len(delay) * cutoffList[idx] / fs))
        # print(f"delay: {np.mean(delay[:cut])}")
        gdMedians.append(np.median(delay[:cut]))

        color = cmap(idx / len(firList))
        for axis in ax[:3]:
            axis.axvline(cutoffList[idx], alpha=0.33, color=color, ls="--")
        ax[0].plot(freq, gain, alpha=0.66, lw=1, color=color)
        ax[1].plot(freq, phase, alpha=0.66, lw=1, color=color)
        ax[2].plot(freq, delay, alpha=0.66, lw=1, color=color)
        ax[3].plot(fir, alpha=0.66, lw=1, color=color, label=f"{nameList[idx]}")

    ax[0].set_ylabel("Gain [dB]")
    ax[0].set_ylim((-40, 6))
    ax[1].set_ylabel("Phase [rad/sample]")
    ax[2].set_ylabel("Delay [sample]")
    gdMid = np.median(gdMedians)
    gdRange = np.max(
        [
            1.5 * (gdMid - np.min(gdMedians)),
            1.5 * (np.max(gdMedians) - gdMid),
            1,
        ]
    )
    ax[2].set_ylim([gdMid - gdRange, gdMid + gdRange])
    ax[3].set_ylabel("FIR Amplitude")
    ax[3].legend(ncol=2)

    for axis in ax[:3]:
        axis.set_xlim([100, 25000])
        axis.set_xscale("log")
        axis.axvline(fs / 2, color="black", ls="--")

    for axis in ax:
        axis.grid(color="#f0f0f0", which="both")
        # axis.legend(ncol=2)
        # axis.set_xscale("log")
    fig.set_size_inches((8, 8))
    fig.tight_layout()
    plt.show()


def plotSpectrogram(sampleRateHz: float, sig: np.ndarray):
    frameSize = 1024
    win = signal.get_window("hann", frameSize)
    sft = signal.ShortTimeFFT(
        win, len(win) // 128, fs=sampleRateHz, scale_to="magnitude"
    )
    mag = sft.stft(sig)

    plt.figure(figsize=(10, 5))
    im1 = plt.imshow(
        np.log(abs(mag) + 1e-7),
        origin="lower",
        aspect="auto",
        extent=sft.extent(len(sig)),
        cmap="magma",
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    # plt.ylim([10, 20000])
    # plt.yscale("log")
    # plt.colorbar(im1, label="Magnitude $|S_x(t, f)|$")
    plt.tight_layout()


def saveSpectrogram(path: Path, sampleRateHz: float, sig: np.ndarray):
    plotSpectrogram(sampleRateHz, sig)
    plt.savefig(path)
    plt.close()


def versinc3(x, fc):
    u = fc * x
    theta = np.pi * u
    if abs(theta) < 0.05 * np.pi:
        t2 = theta * theta
        poly = -2.0 / 467775.0
        poly = poly * t2 + 2.0 / 14175.0
        poly = poly * t2 - 1.0 / 315.0
        poly = poly * t2 + 2.0 / 45.0
        poly = poly * t2 - 1.0 / 3.0
        poly = poly * t2 + 1.0
        return (2 * fc * theta) * poly

    sn = np.sin(np.pi * (u - np.round(u)))
    return 2 * sn * sn / (np.pi * x)


def versinc(x, fc):
    """
    All-pass filter that works as Hilbert transformer.

    Return value is `(1 - cos(2*pi*fc*x)/(fc*x))`.

    It may not work when `fc < 1 / N` where `N` is the length of the filter.
    """
    x = np.asarray(x, dtype=np.float64)
    u = fc * x
    theta = np.pi * u

    out = np.zeros_like(x)
    mask = np.abs(theta) < 0.05 * np.pi

    if np.any(mask):
        theta_small = theta[mask]
        t2 = theta_small * theta_small

        poly = -2.0 / 467775.0
        poly = poly * t2 + 2.0 / 14175.0
        poly = poly * t2 - 1.0 / 315.0
        poly = poly * t2 + 2.0 / 45.0
        poly = poly * t2 - 1.0 / 3.0
        poly = poly * t2 + 1.0

        out[mask] = (2 * fc * theta_small) * poly

    if np.any(~mask):
        u_large = u[~mask]
        x_large = x[~mask]
        sn = np.sin(np.pi * (u_large - np.round(u_large)))
        out[~mask] = 2 * sn * sn / (np.pi * x_large)

    return out


def triangleC(length: int, fraction: float):
    radius = (length + 2) / 2.0
    center = length / 2 - fraction
    window = np.zeros(length)
    for i in range(length):
        val = 1.0 - abs(i - center) / radius
        window[i] = max(0.0, val)
    return window


def blackmanHarrisC(length: int, fraction: float):
    isEven = 1 - length % 2

    ω = 2 * np.pi / float(length - isEven)
    k = 2 * np.cos(ω)
    u1 = np.cos(ω * (-1.0 + fraction))
    u2 = np.cos(ω * (-2.0 + fraction))

    window = np.zeros(length)
    for i in range(length):
        u0 = k * u1 - u2
        u2 = u1
        u1 = u0
        window[i] = 0.21747 + u0 * (-0.45325 + u0 * (0.28256 + u0 * -0.04672))
    return window


def lowpassWindowedReversed(length: int, cutoff: float, fractionSample: float):
    fftbins = True

    mid = fractionSample - (length // 2 + length % 2)

    fir = np.zeros(length)
    for i in range(length):
        fir[i] = versinc(i + mid, cutoff)
    # return fir * triangleC(length, fractionSample)
    # return fir * blackmanHarrisC(length, fractionSample)
    return fir
    # return fir * signal.get_window("blackman", length, fftbins)
    # return fir * signal.get_window("blackmanharris", length, fftbins)
    # return fir * signal.get_window(("chebwin", 120), length, fftbins)
    # return fir * signal.get_window("cosine", length, fftbins)
    # return fir * signal.get_window(("dpss", 4), length, fftbins)
    # return fir * signal.get_window(("exponential", None, 16), length, fftbins)
    # return fir * signal.get_window(("gaussian", length / 9), length, fftbins)
    # return fir * signal.get_window("hann", length, fftbins)
    # return fir * signal.get_window("hamming", length, fftbins)
    # return fir * signal.get_window(("kaiser", 4 * np.pi), length, fftbins)
    # return fir * signal.get_window("nuttall", length, fftbins)
    # return fir * signal.get_window("triang", length, fftbins)


def fastVersinc(length: int, cutoff: float, fractionSample: float):
    mid = fractionSample - (length // 2 + length % 2)

    omega = np.pi * cutoff
    phi = mid * omega
    u1 = np.sin(phi - omega)
    u2 = np.sin(phi - 2 * omega)
    k = 2 * np.cos(omega)

    fir = np.zeros(length)
    for i in range(length):
        u0 = k * u1 - u2
        u2 = u1
        u1 = u0

        x = i + mid
        theta = np.pi * cutoff * x
        if abs(theta) < 0.05 * np.pi:
            t2 = theta * theta
            poly = -2.0 / 467775.0
            poly = poly * t2 + 2.0 / 14175.0
            poly = poly * t2 - 1.0 / 315.0
            poly = poly * t2 + 2.0 / 45.0
            poly = poly * t2 - 1.0 / 3.0
            poly = poly * t2 + 1.0
            fir[i] = (2 * cutoff * theta) * poly
        else:
            fir[i] = 2 * u0 * u0 / (np.pi * x)
    return fir


def testFir():
    sampleRate = 48000
    tap = 256
    fraction = 1 - np.finfo(np.float64).eps

    nOctave = 6  # int(np.round(np.log2(tap)))

    fir = []
    cutoff = []
    name = []
    for idx in range(2, nOctave + 1):
        cutoff.append(48000 / 2**idx)
        name.append(str(cutoff[-1]))
        # fir.append(lowpassWindowedReversed(tap, cutoff[-1] / sampleRate, fraction))
        fir.append(fastVersinc(tap, cutoff[-1] / sampleRate, fraction))
    plotResponse(fir, cutoff, name, fs=sampleRate)


def compareFir():
    sampleRate = 48000
    tap = 256
    cutoff = 0.1
    fraction = 1e-5  # np.finfo(np.float64).eps
    param = [tap, cutoff, fraction]

    fir = []
    name = ["ref.", "fast."]
    fir.append(lowpassWindowedReversed(*param))
    fir.append(fastVersinc(*param))
    print(f"diff: {np.sum(np.abs(fir[0]-fir[1]))}")
    plotResponse(fir, [param[1] * sampleRate] * 2, name, fs=sampleRate)
    plt.show()


def generateSawtooth(
    sampleRate: float, frequencyHz: float, durationSecond: float
) -> np.ndarray:
    """Integer sample period only. It makes easier to check the aliasing on spectrogram."""
    period: int = int(sampleRate / frequencyHz)
    durationSample = int(sampleRate * durationSecond)
    if period == 0:
        return np.zeros(durationSample)
    sig = np.arange(durationSample, dtype=np.int64) % period
    return sig.astype(np.float64) / float(period) * 2 - 1


def testDelay(inputPath: Path | None = None):
    def loadSound(path: Path | None):
        if path is None or not path.exists():
            sampleRate = 48000
            return (sampleRate, generateSawtooth(sampleRate, 1000, 1))
        data, fs = soundfile.read(path, always_2d=True)
        return (fs, data.T[0])

    sampleRate, source = loadSound(inputPath)
    # saveSpectrogram(Path("img/testDelay_source.png"), sampleRate, source)

    # delayTime = np.linspace(8001, 1, len(source))
    # delayTime = 8001 - np.geomspace(1, 8000, len(source))
    # delayTime = np.geomspace(8000, 1, len(source))
    # delayTime = np.full_like(source, 8000)
    delayTime = np.hstack(
        [
            np.linspace(0, 512, len(source) // 4),
            np.linspace(512, 0, len(source) // 4),
            np.linspace(0, 512, len(source) // 4),
            np.linspace(512, 0, len(source) // 4),
        ]
    )
    # delayTime = 200 * 2 ** (8 * np.sin(4 * np.pi * np.arange(len(source)) / sampleRate))
    # delayTime = 16 * 2 ** (4 * np.sin(4 * np.pi * np.arange(len(source)) / sampleRate))

    outputGain = 0.25
    maxDelayTimeSample = 65536
    # for method in [m for m in dir(Delay) if "process" in m]:
    for method in tqdm.tqdm(
        [
            # "processVersinc",
            "processVersincBlackmanHarris",
        ]
    ):
        delay = Delay(maxDelayTimeSample)
        func = getattr(delay, method)
        delayed = np.zeros_like(source)
        for i, v in enumerate(source):
            delayed[i] = func(v, delayTime[i])

        saveSpectrogram(Path(f"img/testDelay_{method}.png"), sampleRate, delayed)

        soundfile.write("snd/source.wav", outputGain * source, sampleRate, "FLOAT")
        soundfile.write(f"snd/{method}.wav", outputGain * delayed, sampleRate, "FLOAT")


if __name__ == "__main__":
    Path("img").mkdir(parents=True, exist_ok=True)
    Path("snd").mkdir(parents=True, exist_ok=True)

    # testFir()
    # compareFir()
    testDelay()
