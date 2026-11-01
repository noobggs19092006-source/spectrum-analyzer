# Wireless Spectrum Analyzer and Modulation Recognition

Final project. See Project_Report.txt and Project_Learning_Guide.pdf.

Not stored in this repository (too large for GitHub). Share separately:
- pluto_frames.h5 (Pluto dataset, 565 MB)
- dataset_sim.h5 and dataset_cfo15.h5 (regenerate with sim/sim_gen.py)
- session1/ and session2/ raw recordings (session*_all.bin)
- overair/fm_93p5_g10.bin (over-the-air recording)

Run order: build_pluto_dataset.py, sim/sim_gen.py, train/train.py,
model/export_onnx.py, eval/evaluate.py. Live demo: start_demo.bat.
