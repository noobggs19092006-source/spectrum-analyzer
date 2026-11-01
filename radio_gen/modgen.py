"""
radio_gen/modgen.py
===================
Shared waveform generator used by BOTH sim_gen.py and pluto_tx.py.

Contract (spec §2 and §3A):
  - fs = 1_600_000 Sa/s, symbol_rate = 200_000 Baud, SPS = 8
  - RRC pulse shaping: alpha = 0.35, span = 8 symbols
  - All randomness from the caller-supplied ``rng`` (numpy.random.Generator)
  - Returns complex64, length exactly n, unit average power

Usage:
    import numpy as np
    from radio_gen.modgen import gen_waveform, CLASS_NAMES

    rng = np.random.default_rng(42)
    iq = gen_waveform("QPSK", 4096, rng)   # complex64, len 4096, |iq|^2 mean ≈ 1
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert as _scipy_hilbert

# ── Fixed constants (do NOT change without updating the contract) ─────────────
FS: int = 1_600_000          # sample rate, Hz
SYMBOL_RATE: int = 200_000   # baud
SPS: int = FS // SYMBOL_RATE  # 8 samples / symbol
RRC_ALPHA: float = 0.35
RRC_SPAN: int = 8            # symbols (filter spans 8*8+1 = 65 taps)

CLASS_NAMES: list[str] = [
    "BPSK", "QPSK", "8PSK", "QAM16", "QAM64",
    "GFSK", "CPFSK", "PAM4", "WBFM", "AMDSB", "AMSSB",
]
CLASS_IDX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}


# ════════════════════════════════════════════════════════════════════════════ #
#  Root-Raised-Cosine filter                                                   #
# ════════════════════════════════════════════════════════════════════════════ #

def _rrc_taps(alpha: float = RRC_ALPHA,
              sps: int = SPS,
              span: int = RRC_SPAN) -> np.ndarray:
    """Energy-normalised RRC impulse response (float32, length span*sps+1)."""
    n = span * sps          # half-length (filter has n+1 taps total)
    t = np.arange(-n // 2, n // 2 + 1) / sps   # time in symbol periods

    h = np.empty_like(t)
    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = 1.0 - alpha + (4.0 * alpha / np.pi)
        elif abs(abs(ti) - 1.0 / (4.0 * alpha)) < 1e-9:
            s = np.sin(np.pi / (4.0 * alpha))
            c = np.cos(np.pi / (4.0 * alpha))
            h[i] = (alpha / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * s + (1.0 - 2.0 / np.pi) * c
            )
        else:
            num = (
                np.sin(np.pi * ti * (1.0 - alpha))
                + 4.0 * alpha * ti * np.cos(np.pi * ti * (1.0 + alpha))
            )
            den = np.pi * ti * (1.0 - (4.0 * alpha * ti) ** 2)
            h[i] = num / den

    h /= np.sqrt(float(np.dot(h, h)))   # energy-normalise
    return h.astype(np.float32)


# Compute once at import time
_RRC: np.ndarray = _rrc_taps()
_RRC_DELAY: int = len(_RRC) // 2   # group delay in samples


# ════════════════════════════════════════════════════════════════════════════ #
#  Constellation helpers                                                        #
# ════════════════════════════════════════════════════════════════════════════ #

def _bpsk_const() -> np.ndarray:
    return np.array([-1.0 + 0j, 1.0 + 0j], dtype=np.complex128)


def _qpsk_const() -> np.ndarray:
    # Gray-coded: bits 00→+1+j, 01→-1+j, 11→-1-j, 10→+1-j (normalised)
    return np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex128) / np.sqrt(2.0)


def _8psk_const() -> np.ndarray:
    # Standard Gray-coded 8-PSK: assign Gray codes to equally-spaced phases
    # so adjacent phases differ by exactly 1 bit.
    gray = [0, 1, 3, 2, 6, 7, 5, 4]   # Gray code for each phase index
    phases = np.array(gray, dtype=float) * 2.0 * np.pi / 8.0
    return np.exp(1j * phases)


def _gray_to_nat(g: int) -> int:
    """Convert a Gray-coded integer to its natural (binary) index."""
    n = 0
    while g:
        n ^= g
        g >>= 1
    return n


def _qam_const(order: int) -> np.ndarray:
    """Gray-coded square QAM constellation, normalised to unit average power."""
    m = int(round(np.sqrt(order)))
    assert m * m == order, f"QAM order must be a perfect square, got {order}"
    bits_per_axis = int(round(np.log2(m)))

    # Amplitude levels per axis: -(m-1), -(m-3), ..., (m-3), (m-1)
    levels = np.arange(-(m - 1), m, 2, dtype=float)

    pts: list[complex] = []
    for sym_idx in range(order):
        i_gray = sym_idx >> bits_per_axis    # upper bits → I axis
        q_gray = sym_idx & (m - 1)           # lower bits → Q axis
        pts.append(levels[_gray_to_nat(i_gray)] + 1j * levels[_gray_to_nat(q_gray)])

    c = np.array(pts, dtype=np.complex128)
    c /= np.sqrt(np.mean(np.abs(c) ** 2))   # unit average power
    return c


def _pam4_const() -> np.ndarray:
    """PAM4: Gray-coded levels {-3,-1,+1,+3}, unit average power, on I axis."""
    # data bits → level: 00→-3, 01→-1, 10→+1, 11→+3 (standard Gray ordering)
    levels = np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(5.0)
    return levels.astype(np.complex128)   # Q = 0


# ════════════════════════════════════════════════════════════════════════════ #
#  Linear-modulation pulse shaper                                              #
# ════════════════════════════════════════════════════════════════════════════ #

def _n_symbols_needed(n: int) -> int:
    """Symbols required so that trimming [_RRC_DELAY : _RRC_DELAY+n] is safe."""
    return int(np.ceil((_RRC_DELAY + n) / SPS)) + 2   # +2 guard


def _pulse_shape(symbols: np.ndarray, n: int) -> np.ndarray:
    """Upsample symbols by SPS and apply RRC filter; return exactly n samples."""
    upsampled = np.zeros(len(symbols) * SPS, dtype=np.complex64)
    upsampled[::SPS] = symbols.astype(np.complex64)
    filtered = np.convolve(upsampled, _RRC.astype(np.complex64))
    start = _RRC_DELAY
    end = start + n
    if end > len(filtered):
        filtered = np.concatenate(
            [filtered, np.zeros(end - len(filtered), dtype=np.complex64)]
        )
    return filtered[start:end].astype(np.complex64)


# ════════════════════════════════════════════════════════════════════════════ #
#  Per-class waveform generators                                               #
# ════════════════════════════════════════════════════════════════════════════ #

def _gen_bpsk(n: int, rng: np.random.Generator) -> np.ndarray:
    c = _bpsk_const()
    syms = c[rng.integers(0, len(c), _n_symbols_needed(n))]
    return _pulse_shape(syms, n)


def _gen_qpsk(n: int, rng: np.random.Generator) -> np.ndarray:
    c = _qpsk_const()
    syms = c[rng.integers(0, len(c), _n_symbols_needed(n))]
    return _pulse_shape(syms, n)


def _gen_8psk(n: int, rng: np.random.Generator) -> np.ndarray:
    c = _8psk_const()
    syms = c[rng.integers(0, len(c), _n_symbols_needed(n))]
    return _pulse_shape(syms, n)


def _gen_qam16(n: int, rng: np.random.Generator) -> np.ndarray:
    c = _qam_const(16)
    syms = c[rng.integers(0, len(c), _n_symbols_needed(n))]
    return _pulse_shape(syms, n)


def _gen_qam64(n: int, rng: np.random.Generator) -> np.ndarray:
    c = _qam_const(64)
    syms = c[rng.integers(0, len(c), _n_symbols_needed(n))]
    return _pulse_shape(syms, n)


def _gen_pam4(n: int, rng: np.random.Generator) -> np.ndarray:
    c = _pam4_const()
    syms = c[rng.integers(0, len(c), _n_symbols_needed(n))]
    return _pulse_shape(syms, n)


def _gen_gfsk(n: int, rng: np.random.Generator,
              bt: float = 0.5, h: float = 0.5) -> np.ndarray:
    """GFSK: Gaussian BT=0.5, modulation index h=0.5."""
    from scipy.signal.windows import gaussian

    guard = 4 * SPS                                   # pre-roll samples
    n_total = n + guard
    n_sym = int(np.ceil(n_total / SPS)) + 4           # +4 for Gaussian tail

    bits = rng.integers(0, 2, n_sym)
    nrz = (2.0 * bits - 1.0).astype(np.float64)      # ±1

    # Upsample NRZ by SPS
    freq_up = np.repeat(nrz, SPS)

    # Gaussian shaping filter (4-symbol span)
    gauss_span = 4
    n_taps = gauss_span * SPS + 1
    sigma = SPS * np.sqrt(np.log(2.0)) / (2.0 * np.pi * bt)
    g = gaussian(n_taps, std=sigma)
    g /= g.sum()                                      # area = 1

    freq_filt = np.convolve(freq_up, g, mode="same")

    # Integrate → instantaneous phase; h*π radians per symbol
    phase = np.cumsum(freq_filt) * (np.pi * h / SPS)
    sig = np.exp(1j * phase).astype(np.complex64)

    # Drop pre-roll and return exactly n samples
    return sig[guard: guard + n]


def _gen_cpfsk(n: int, rng: np.random.Generator, h: float = 0.5) -> np.ndarray:
    """CPFSK: rectangular frequency pulse, modulation index h=0.5."""
    guard = SPS
    n_sym = int(np.ceil((n + guard) / SPS)) + 2
    bits = rng.integers(0, 2, n_sym)
    nrz = (2.0 * bits - 1.0).astype(np.float64)

    freq_up = np.repeat(nrz, SPS)                    # rectangular pulse
    phase = np.cumsum(freq_up) * (np.pi * h / SPS)
    sig = np.exp(1j * phase).astype(np.complex64)
    return sig[guard: guard + n]


def _gen_wbfm(n: int, rng: np.random.Generator,
              deviation: float = 75e3,
              audio_bw: float = 15e3) -> np.ndarray:
    """WBFM: sum-of-tones audio < 15 kHz, frequency deviation 75 kHz."""
    t = np.arange(n, dtype=np.float64) / FS

    n_tones = int(rng.integers(3, 8))
    freqs = rng.uniform(100.0, audio_bw, n_tones)
    phases = rng.uniform(0.0, 2.0 * np.pi, n_tones)
    amps = rng.uniform(0.3, 1.0, n_tones)

    audio = sum(a * np.sin(2.0 * np.pi * f * t + p)
                for a, f, p in zip(amps, freqs, phases))
    audio /= max(float(np.max(np.abs(audio))), 1e-9)   # normalise to ±1

    # FM: φ(t) = 2π · kf · ∫ audio(τ) dτ
    phase = 2.0 * np.pi * deviation / FS * np.cumsum(audio)
    return np.exp(1j * phase).astype(np.complex64)


def _gen_amdsb(n: int, rng: np.random.Generator,
               mu: float = 0.8,
               audio_bw: float = 3400.0) -> np.ndarray:
    """AM-DSB: carrier with modulation index 0.8, voice-like audio."""
    t = np.arange(n, dtype=np.float64) / FS

    n_tones = int(rng.integers(1, 4))
    freqs = rng.uniform(300.0, audio_bw, n_tones)
    phases = rng.uniform(0.0, 2.0 * np.pi, n_tones)

    audio = sum(np.sin(2.0 * np.pi * f * t + p)
                for f, p in zip(freqs, phases))
    audio /= max(float(np.max(np.abs(audio))), 1e-9)

    # Baseband DSB-LC: (1 + μ·m(t)), real signal → complex with Q=0
    return (1.0 + mu * audio).astype(np.complex64)


def _gen_amssb(n: int, rng: np.random.Generator, mu: float = 0.8) -> np.ndarray:
    """AM-SSB: Hilbert-transform the DSB baseband to keep upper sideband only."""
    dsb = _gen_amdsb(n, rng, mu)
    # scipy.signal.hilbert returns the analytic signal: real part = original,
    # imag part = Hilbert transform → this IS the upper-sideband complex envelope.
    analytic = _scipy_hilbert(dsb.real)
    return analytic.astype(np.complex64)


# ════════════════════════════════════════════════════════════════════════════ #
#  Dispatch table                                                               #
# ════════════════════════════════════════════════════════════════════════════ #

_GENERATORS: dict[str, object] = {
    "BPSK":  _gen_bpsk,
    "QPSK":  _gen_qpsk,
    "8PSK":  _gen_8psk,
    "QAM16": _gen_qam16,
    "QAM64": _gen_qam64,
    "GFSK":  _gen_gfsk,
    "CPFSK": _gen_cpfsk,
    "PAM4":  _gen_pam4,
    "WBFM":  _gen_wbfm,
    "AMDSB": _gen_amdsb,
    "AMSSB": _gen_amssb,
}


# ════════════════════════════════════════════════════════════════════════════ #
#  Public API                                                                   #
# ════════════════════════════════════════════════════════════════════════════ #

def gen_waveform(
    class_name: str,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a baseband IQ waveform for the requested modulation class.

    Parameters
    ----------
    class_name : str
        One of the 11 class names (case-sensitive): BPSK QPSK 8PSK QAM16
        QAM64 GFSK CPFSK PAM4 WBFM AMDSB AMSSB.
    n : int
        Number of output samples at fs = 1_600_000 Sa/s.
    rng : numpy.random.Generator
        Seeded random generator.  Pass ``numpy.random.default_rng(seed)``
        for full reproducibility.

    Returns
    -------
    numpy.ndarray
        Complex64 array of length ``n``, unit average power
        (mean(|x|^2) ≈ 1.0).
    """
    if class_name not in _GENERATORS:
        raise ValueError(
            f"Unknown class '{class_name}'.  Valid names: {CLASS_NAMES}"
        )

    sig = _GENERATORS[class_name](n, rng)     # type: ignore[operator]

    if len(sig) != n:
        raise RuntimeError(
            f"Generator for '{class_name}' returned {len(sig)} samples, expected {n}."
        )

    # Unit average power normalisation
    pwr = float(np.mean(np.abs(sig) ** 2))
    if pwr > 0.0:
        sig = sig / np.sqrt(pwr)

    return sig.astype(np.complex64)
