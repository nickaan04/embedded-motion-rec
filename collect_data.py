#!/usr/bin/env python3
"""Host-side motion dataset collector for embedded-motion-rec.

Workflow (one motion label per upload session):
  1. Set CURRENT_LABEL in embedded-motion-rec.ino and upload to the board.
  2. Set COLLECTION_LABEL below to the same string and run this script.
  3. When you see "Ready", press SPACEBAR.
  4. A short countdown runs, then the board records 1 second of IMU data.
  5. Rows are appended to data.csv; repeat from step 3 until done.
  6. Change the label in the .ino, re-upload, update COLLECTION_LABEL, run again.

Press q (instead of space) to quit.
"""

from __future__ import annotations

import argparse
import csv
import sys
import termios
import time
import tty
from pathlib import Path

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("Install dependencies: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

# CHANGE THIS TO MATCH CURRENT_LABEL IN THE .INO
# Options: "still", "shake", "tilt left", "tilt right",
#          "tilt forward", "tilt backward"
COLLECTION_LABEL = "tilt forward"

# Seconds to count down after SPACEBAR before the board starts sampling.
COUNTDOWN_SECONDS = 3

LABEL_HINTS: dict[str, str] = {
    "still": "Hold the board steady.",
    "shake": "Shake the board back and forth.",
    "tilt left": "Tilt the board to your left.",
    "tilt right": "Tilt the board to your right.",
    "tilt forward": "Tilt the top of the board toward you.",
    "tilt backward": "Tilt the top of the board away from you.",
}

CSV_HEADER = ["sample_id", "timestamp", "accel_x", "accel_y", "accel_z", "label"]
SAMPLES_PER_WINDOW = 50


def label_slug(label: str) -> str:
    """Turn a label into the sample_id prefix (spaces -> underscores)."""
    return label.replace(" ", "_")


def make_sample_id(label: str, sample_number: int) -> str:
    """Naming convention: <label>_<n>, e.g. tilt_left_0, still_12."""
    return f"{label_slug(label)}_{sample_number}"


def find_port(hint: str | None) -> str:
    ports = list(list_ports.comports())
    if hint:
        for p in ports:
            if hint in (p.device, p.name, p.description):
                return p.device
        raise SystemExit(f"No serial port matching '{hint}'")
    for p in ports:
        text = f"{p.description} {p.manufacturer} {p.device}".lower()
        if "nano 33 ble" in text or "nano33ble" in text:
            return p.device
    if len(ports) == 1:
        return ports[0].device
    if not ports:
        raise SystemExit("No serial ports found. Connect the board and try again.")
    raise SystemExit(
        "Multiple serial ports found; pass --port explicitly:\n"
        + "\n".join(f"  {p.device}  ({p.description})" for p in ports)
    )


def next_sample_number(csv_path: Path, label: str) -> int:
    """Find the next index N for sample_ids like tilt_left_N for this label."""
    prefix = label_slug(label) + "_"
    max_n = -1

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return 0

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return 0
        for row in reader:
            sid = row.get("sample_id", "")
            if not sid.startswith(prefix):
                continue
            suffix = sid[len(prefix) :]
            if suffix.isdigit():
                max_n = max(max_n, int(suffix))

    return max_n + 1


def ensure_trailing_newline(csv_path: Path) -> None:
    """Append a newline if the file exists but does not end with one."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return
    with csv_path.open("rb") as f:
        f.seek(-1, 2)
        if f.tell() == 0:
            return
        last_byte = f.read(1)
    if last_byte != b"\n":
        with csv_path.open("ab") as f:
            f.write(b"\n")


def ensure_csv_header(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)
        return
    # Existing file (e.g. hand-edited header) must end with newline before append.
    ensure_trailing_newline(csv_path)


def open_board_serial(port: str, baud: int) -> serial.Serial:
    """Open the USB port and trigger a clean board reset (Nano 33 BLE)."""
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = 0.1
    ser.dtr = False
    ser.rts = False
    ser.open()
    time.sleep(0.3)
    ser.reset_input_buffer()
    # Pulse DTR to reset the board so setup() runs after we are listening.
    ser.dtr = True
    time.sleep(0.3)
    return ser


def try_read_line(ser: serial.Serial, timeout: float = 0.5) -> str | None:
    """Read one line, or return None if nothing arrives before timeout."""
    old_timeout = ser.timeout
    ser.timeout = min(old_timeout, 0.1) if old_timeout else 0.1
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            try:
                raw = ser.readline()
            except serial.SerialException:
                return None
            if raw:
                return raw.decode("utf-8", errors="replace").strip()
        return None
    finally:
        ser.timeout = old_timeout


def read_line(ser: serial.Serial, timeout: float = 10.0) -> str:
    """Read one line; raise if the board does not respond in time."""
    line = try_read_line(ser, timeout=timeout)
    if line is None:
        raise TimeoutError("Timed out waiting for board response")
    return line


def wait_for_spacebar() -> None:
    """Block until the user presses space (or q to quit)."""
    print("  Press SPACEBAR when you are ready to record  (q to quit)")
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == " ":
                print("  SPACEBAR — starting countdown...")
                return
            if ch in ("q", "Q", "\x03"):  # q or Ctrl-C
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def run_countdown(seconds: int) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"    {remaining}...")
        time.sleep(1)
    print("    GO — perform the motion now!")


def record_window(ser: serial.Serial, sample_id: str) -> list[list[str]]:
    ser.write(f"R {sample_id}\n".encode())
    ser.flush()

    if read_line(ser) != "RECORDING":
        raise RuntimeError("Board did not start recording")

    rows: list[list[str]] = []
    for _ in range(SAMPLES_PER_WINDOW):
        line = read_line(ser, timeout=10.0)
        parts = line.split(",")
        if len(parts) != 6:
            raise RuntimeError(f"Bad sample line ({len(parts)} fields): {line!r}")
        rows.append(parts)

    if read_line(ser) != "DONE":
        raise RuntimeError("Board did not finish recording")

    if len(rows) != SAMPLES_PER_WINDOW:
        raise RuntimeError(f"Expected {SAMPLES_PER_WINDOW} samples, got {len(rows)}")
    if rows[0][1] != "0":
        raise RuntimeError(f"First timestamp should be 0, got {rows[0][1]!r}")
    if rows[-1][1] != str((SAMPLES_PER_WINDOW - 1) * 20):
        raise RuntimeError(f"Last timestamp should be 980, got {rows[-1][1]!r}")

    return rows


def append_rows(csv_path: Path, rows: list[list[str]]) -> None:
    ensure_trailing_newline(csv_path)
    with csv_path.open("a", newline="") as f:
        csv.writer(f).writerows(rows)


def ping_board(ser: serial.Serial) -> None:
    ser.write(b"PING\n")
    ser.flush()


def wait_for_board_startup(ser: serial.Serial, expected_label: str) -> None:
    """Wait for READY + LABEL lines after the board reboots (can take several seconds)."""
    print("  Waiting for board to boot (up to ~20 s)...")

    deadline = time.monotonic() + 20.0
    board_label = None
    saw_ready = False
    last_ping = 0.0

    while time.monotonic() < deadline:
        if time.monotonic() - last_ping > 1.0:
            ping_board(ser)
            last_ping = time.monotonic()

        line = try_read_line(ser, timeout=0.4)
        if line is None:
            continue
        print(f"  Board: {line}")
        if line == "READY":
            saw_ready = True
        elif line.startswith("LABEL "):
            board_label = line[6:].strip()
        if saw_ready and board_label is not None:
            break

    if not saw_ready or board_label is None:
        raise SystemExit(
            "\nCould not talk to the Arduino. Try:\n"
            "  1. Re-upload embedded-motion-rec.ino\n"
            "  2. Close Serial Monitor (115200 baud in IDE if you test there)\n"
            "  3. Wait 2 s, then run: python3 collect_data.py --port /dev/cu.usbmodem141301\n"
            "  4. List ports: python3 collect_data.py --list-ports"
        )

    if board_label != expected_label:
        raise SystemExit(
            f"\nLabel mismatch: sketch reports '{board_label}' but "
            f"COLLECTION_LABEL = '{expected_label}'.\n"
            "Set both to the same value, re-upload the .ino, and run again."
        )

    # Drain leftover startup text (RATE line, etc.)
    while True:
        line = try_read_line(ser, timeout=0.2)
        if line is None:
            break
        print(f"  Board: {line}")


def run_session(port: str, csv_path: Path, label: str, baud: int) -> None:
    ensure_csv_header(csv_path)
    sample_number = next_sample_number(csv_path, label)
    hint = LABEL_HINTS.get(label, "")

    print(f"\nCollecting label: '{label}'")
    if hint:
        print(f"  Motion: {hint}")
    print(f"  sample_id format: {label_slug(label)}_<n>  (next: {make_sample_id(label, sample_number)})")
    print(f"  Output file: {csv_path.resolve()}\n")

    ser = open_board_serial(port, baud)
    try:
        # Let setup() finish (IMU init + serial prints) after reset.
        time.sleep(2)
        wait_for_board_startup(ser, label)

        print("\n" + "=" * 60)
        print("  READY — press SPACEBAR.")
        print("  (Click the terminal first so it has focus. Press q to quit.)")
        print("=" * 60)

        window_count = 0
        while True:
            sample_id = make_sample_id(label, sample_number)
            print(f"\n--- Window #{window_count + 1}  (sample_id: {sample_id}) ---")

            try:
                wait_for_spacebar()
            except KeyboardInterrupt:
                break

            run_countdown(COUNTDOWN_SECONDS)

            rows = record_window(ser, sample_id)
            append_rows(csv_path, rows)

            window_count += 1
            sample_number += 1
            print(f"  Saved 50 rows as '{sample_id}' -> {csv_path.name}")
            print("\n" + "=" * 60)
            print("  READY — press SPACEBAR in THIS terminal for the next window.")
            print("=" * 60)

        print(f"\nStopped. {window_count} windows saved for '{label}'.")
    finally:
        ser.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect IMU windows for one label (spacebar + countdown)"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).parent / "data.csv",
        help="Output CSV path (default: data.csv in project folder)",
    )
    parser.add_argument(
        "--port",
        help="Serial port (e.g. /dev/cu.usbmodem1101). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Serial baud rate (must match sketch)",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List serial ports and exit",
    )
    args = parser.parse_args()

    if args.list_ports:
        for p in list_ports.comports():
            print(f"{p.device}\t{p.description}")
        return

    port = find_port(args.port)
    print(f"Using port {port} at {args.baud} baud")

    try:
        run_session(port, args.csv, COLLECTION_LABEL, args.baud)
    except KeyboardInterrupt:
        print("\nQuit.")


if __name__ == "__main__":
    main()
