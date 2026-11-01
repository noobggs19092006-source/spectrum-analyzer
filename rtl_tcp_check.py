"""rtl_tcp_check.py - read one second of live samples from rtl_tcp."""
import socket
import struct

import numpy as np

s = socket.create_connection(("127.0.0.1", 1234))


def cmd(c, v):
    s.sendall(struct.pack(">BI", c, v))


cmd(0x01, 915_000_000)  # center frequency
cmd(0x02, 2_400_000)    # sample rate
cmd(0x03, 1)            # manual gain
cmd(0x04, 297)          # 29.7 dB, in tenths

need = 2_400_000 * 2    # one second of uint8 I/Q
buf = bytearray()
while len(buf) < need:
    buf += s.recv(65536)

raw = np.frombuffer(bytes(buf[:need]), dtype=np.uint8).astype(np.float32)
iq = ((raw[0::2] - 127.5) + 1j * (raw[1::2] - 127.5)) / 127.5
print("samples:", len(iq), " power (dB):", round(10 * np.log10(np.mean(np.abs(iq) ** 2)), 1))
