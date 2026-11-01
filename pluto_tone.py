import winiio  # must come first: sets up the libiio DLL path
import numpy as np
import adi

FS, LO, N = 1_600_000, 915_000_000, 16384

sdr = adi.Pluto(uri="ip:192.168.2.1")
try:
    sdr.sample_rate = FS
    sdr.tx_lo = LO
    sdr.tx_rf_bandwidth = 2_000_000
    sdr.tx_hardwaregain_chan0 = -40
    sdr.tx_cyclic_buffer = True

    t = np.arange(N) / FS
    sdr.tx(0.5 * np.exp(2j * np.pi * 100_000 * t) * 2**14)

    input("Transmitting a 100 kHz tone at -40 dB. Press Enter to stop...")
finally:
    sdr.tx_destroy_buffer()
    sdr.close()