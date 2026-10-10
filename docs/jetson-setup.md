# Jetson setup checklist

The exact OS, camera pipeline, and compatible ML packages need to be recorded on the target Jetson before hardware commands are standardized. Do not copy unverified CSI commands into the live demo workflow.

## Record the environment

- [ ] JetPack version: `____________`
- [ ] Ubuntu/L4T version: `____________`
- [ ] Python version: `____________`
- [ ] CUDA version: `____________`
- [ ] PyTorch version and source: `____________`
- [ ] TensorRT availability (not required for MVP): `____________`
- [ ] SSH hostname/IP and access method: `____________`

## Repository and base path

- [ ] Clone the repository on the Jetson.
- [ ] Create a virtual environment appropriate to the installed Python version.
- [ ] Install the base package, then only the adapters needed for the next test.
- [ ] Run `PYTHONPATH=src python -m unittest discover -s tests -v`.
- [ ] Run `PYTHONPATH=src python -m thirdeye.main --frames 30` to verify the dependency-free mock path.

## One-camera milestone

- [ ] Connect one IMX219 camera and confirm that the Jetson sees it.
- [ ] Verify a stable live preview with a known-good tool or documented vendor example.
- [ ] Record the exact working OpenCV/GStreamer source string or camera device setup here: `____________`
- [ ] Put that value in ignored `config/local.json`; leave `config/default.json` hardware-neutral.
- [ ] Confirm the stream remains stable for several minutes.

## Local YOLO milestone

- [ ] Choose a lightweight pretrained model compatible with the installed Jetson stack.
- [ ] Record model name, source URL, license, and local cached path: `____________`
- [ ] Install the necessary local package(s), then set `detector.backend` to `ultralytics` and its configured `model` path.
- [ ] Verify person detections and measured FPS on one live feed.
- [ ] Only consider TensorRT export after this baseline works reliably.

## Remote workflow

Use SSH for code changes and logs, but keep the actual camera/display test observable locally at the Jetson during bring-up. Retain the mock and video paths as fallbacks for software debugging; they should not replace the live demo.
