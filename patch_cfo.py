import pathlib
p = pathlib.Path(r"C:\spectrum_rf\build_pluto_dataset.py")
s = p.read_text(encoding="utf-8")
anchor = 'BASE = pathlib.Path(r"C:\\spectrum_rf")'
assert s.count(anchor) == 1, "BASE line not found"
s = s.replace(anchor, anchor + "\nCARRIER_OFFSET_HZ = -12000.0   # measured AM-DSB carrier offset (carrier_check.py)")
old = "y_res = resample_poly(iq_slice(raw, a, b), 2, 3)"
assert s.count(old) == 1, "resample line not found"
new = ("xs = iq_slice(raw, a, b)\n"
       "            t = np.arange(len(xs)) / FS_RTL\n"
       "            xs = (xs * np.exp(-2j * np.pi * CARRIER_OFFSET_HZ * t)).astype(np.complex64)\n"
       "            y_res = resample_poly(xs, 2, 3)")
s = s.replace(old, new)
p.write_text(s, encoding="utf-8")
print("patched")
