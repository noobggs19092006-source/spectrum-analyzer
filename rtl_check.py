import winiio  # must be first: sets up DLL paths
import numpy as np
from rtlsdr import RtlSdr

sdr = RtlSdr()
try:
    sdr.sample_rate = 2.4e6
    sdr.center_freq = 915e6
    sdr.gain = 'auto'
    samples = sdr.read_samples(256 * 1024)
    print("Got", len(samples), "samples")
    print("Mean power (dB):", 10 * np.log10(np.mean(np.abs(samples) ** 2)))
finally:
    sdr.close()