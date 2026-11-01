import pathlib
p = pathlib.Path(r"C:\spectrum_rf\train\train.py")
s = p.read_text(encoding="utf-8")
n_rows = s.count('self.hf["X"][row]')
s = s.replace('self.hf["X"][row]', 'self.X[row]')
s = s.replace('self.hf["y"][row]', 'self.y[row]')
anchor = "        base_idx = np.where(valid_mask)[0]\n"
assert s.count(anchor) == 1, "anchor not found exactly once"
s = s.replace(anchor, anchor + '        # Load arrays into RAM once; per-row reads from compressed HDF5 are very slow\n        self.X = self.hf["X"][:]\n        self.y = self.hf["y"][:]\n')
old = "IQDataset(pluto_path, pluto_session=1)"
assert s.count(old) == 1, "Pluto line not found exactly once"
s = s.replace(old, "IQDataset(pluto_path, pluto_session=1, skip_nan_snr=False)")
p.write_text(s, encoding="utf-8")
print("patched: row reads changed =", n_rows)
