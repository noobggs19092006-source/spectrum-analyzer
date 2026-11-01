import winiio  # must come first: sets up the libiio DLL path
import adi

sdr = adi.Pluto(uri="ip:192.168.2.1")
try:
    sdr.sample_rate = 1600000
    sdr.tx_lo = 915000000
    sdr.tx_rf_bandwidth = 2000000
    print("Sample rate:", sdr.sample_rate)
    print("TX LO:", sdr.tx_lo)
    print("Pluto is ready.")
finally:
    sdr.close()
