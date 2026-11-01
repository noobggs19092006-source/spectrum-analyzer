"""frame_listen.py - prints what the frame cutter publishes (test subscriber)."""
import json

import zmq

ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect("tcp://127.0.0.1:5556")
sub.setsockopt(zmq.SUBSCRIBE, b"frames")

n = 0
while True:
    topic, hdr, payload = sub.recv_multipart()
    h = json.loads(hdr)
    n += 1
    if n == 1 or n % 500 == 0:
        print(f"frame {n}: {h['center_hz']/1e6:.4f} MHz  {h['power_dbfs']:6.1f} dBFS  "
              f"payload {len(payload)} bytes (expect 8192)")
