# CPE 470 Final Project

### Nickaan Jahadi and Michael Drobeck  

This project aims to create an embedded ML motion-recognition game using the Arduino Nano 33 BLE Sense’s onboard IMU. The system will randomly prompt a user to perform a specific physical motion, such as shaking the board, tilting it left, tilting it right, tilting it forward, tilting it backward, or holding it still. Each motion will be represented by a unique LED color, allowing the board to communicate the requested action without needing any external hardware. After the prompt, the user performs the motion, and an on-device ML model classifies the accelerometer data to determine whether the correct action was completed. If single-motion classification works reliably, the system can be extended to recognize short sequences of motions, increasing the difficulty and making the game more engaging.

## Data collection

Training data is stored in `data.csv` (one row per accelerometer sample). Each **window** is 50 samples at 50 Hz (1 second). All 50 rows share the same `sample_id` and `label`.

`sample_id` format: `<label>_<n>`

| Label | LED color (onboard RGB) |
|-------|-------------------------|
| tilt right | red |
| tilt left | green |
| tilt forward | blue |
| tilt backward | magenta |
| shake | yellow |
| still | cyan |

### Workflow

1. Install host dependencies: `pip install -r requirements.txt`
2. Set `CURRENT_LABEL` in `embedded-motion-rec.ino` and upload.
3. Set `COLLECTION_LABEL` in `collect_data.py` to the **same** string.
4. **Close Serial Monitor** in Arduino IDE (only one program can use the USB port).
5. In **Terminal**, run: `python3 collect_data.py`
6. Wait until you see **READY — press SPACEBAR**.
7. Click that terminal so it has focus, then press **SPACEBAR** → 3-2-1 countdown → perform the motion during the 1 s capture.
6. Repeat until you have enough windows for that label. Press **q** to quit.
7. Change the label in the `.ino`, re-upload, update `COLLECTION_LABEL`, and run again.

Use `--port /dev/cu.usbmodemXXXX` if auto-detection picks the wrong device.

