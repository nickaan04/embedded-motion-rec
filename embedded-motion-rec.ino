/*
 * You collect one motion label at a time:
 *   - Change CURRENT_LABEL below to the motion you want (e.g. "tilt left").
 *   - Upload this sketch.
 *   - Run collect_data.py on your computer (set the same label there).
 *   - Press SPACEBAR for each window until you have enough samples.
 *   - Change CURRENT_LABEL, re-upload, and repeat for the next motion.
 *
 * Serial protocol (115200 baud, host script sends these lines):
 *   R <sample_id>   Start recording one window, e.g. "R still_3"
 *   ?             Print help text
 * */

#include "Arduino_LSM9DS1.h"

// CHANGE THIS FOR EACH COLLECTION SESSION
// Must match COLLECTION_LABEL in collect_data.py on your computer
// Valid values: "still", "shake", "tilt left", "tilt right", "tilt forward", "tilt backward"
const char CURRENT_LABEL[] = "tilt forward";

// Sampling configuration
// One training example = 1 second of accelerometer data at 50 Hz = 50 rows.
static const uint8_t SAMPLES_PER_WINDOW = 50;
static const uint16_t SAMPLE_INTERVAL_MS = 20;  // 1000 ms / 50 = 20 ms between samples

// Set true while a window is being captured (ignore new serial commands).
static bool recording = false;

// RGB LED helpers
// The Nano 33 BLE Sense RBG LED is ACTIVE-LOW

void setRgb(bool redOn, bool greenOn, bool blueOn) {
  digitalWrite(LEDR, redOn ? LOW : HIGH);
  digitalWrite(LEDG, greenOn ? LOW : HIGH);
  digitalWrite(LEDB, blueOn ? LOW : HIGH);
}

void ledOff() {
  setRgb(false, false, false);
}

// Show the LED color that matches the motion label (see README color table).
void showLabelLed(const char *label) {
  if (strcmp(label, "tilt right") == 0) {
    setRgb(true, false, false);       // red
  } else if (strcmp(label, "tilt left") == 0) {
    setRgb(false, false, true);     // green
  } else if (strcmp(label, "tilt forward") == 0) {
    setRgb(false, true, false);     // blue
  } else if (strcmp(label, "tilt backward") == 0) {
    setRgb(true, true, false);      // magenta (red + blue)
  } else if (strcmp(label, "shake") == 0) {
    setRgb(true, false, true);      // yellow (red + green)
  } else if (strcmp(label, "still") == 0) {
    setRgb(false, true, true);      // cyan (green + blue)
  } else {
    ledOff();                       // unknown label — LED off
  }
}

// ----------------------------------------------------------------------------
// Accelerometer
// Values are in g (1.0 ≈ gravity). We poll whenever a new sample is ready.
// ----------------------------------------------------------------------------

bool readAccel(float &x, float &y, float &z) {
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(x, y, z);
    return true;
  }
  return false;
}

// Capture exactly 50 samples over 1000 ms, then print CSV lines to Serial.
// Format per row: sample_id,timestamp_ms,accel_x,accel_y,accel_z,label
void recordWindow(const char *sampleId) {
  float x = 0, y = 0, z = 0;
  readAccel(x, y, z);  // prime with the latest reading

  recording = true;
  Serial.println("RECORDING");

  // Anchor all sample times to the start of this window.
  uint32_t windowStart = millis();

  for (uint8_t i = 0; i < SAMPLES_PER_WINDOW; i++) {
    // Wait until this sample's scheduled time (0, 20, 40, ... 980 ms).
    uint32_t targetMs = windowStart + (uint32_t)i * SAMPLE_INTERVAL_MS;
    while ((int32_t)(millis() - targetMs) < 0) {
      readAccel(x, y, z);  // keep IMU fresh while waiting
    }

    readAccel(x, y, z);

    Serial.print(sampleId);
    Serial.print(',');
    Serial.print(i * SAMPLE_INTERVAL_MS);  // timestamp within the window (ms)
    Serial.print(',');
    Serial.print(x, 6);
    Serial.print(',');
    Serial.print(y, 6);
    Serial.print(',');
    Serial.print(z, 6);
    Serial.print(',');
    Serial.println(CURRENT_LABEL);
  }

  Serial.println("DONE");
  recording = false;
  showLabelLed(CURRENT_LABEL);  // restore cue LED after capture
}

// Parse one line from the USB serial port (from collect_data.py).
void handleCommand(const char *line) {
  if (strncmp(line, "R ", 2) == 0) {
    // Example: "R still_0" — host assigns the sample_id string.
    recordWindow(line + 2);
  } else if (strcmp(line, "?") == 0) {
    Serial.println("Commands: R <sample_id> | P | ?");
    Serial.print("Current label: ");
    Serial.println(CURRENT_LABEL);
  } else if (strcmp(line, "P") == 0 || strcmp(line, "PING") == 0) {
    // Host can ping after connect in case it missed the boot messages.
    Serial.println("READY");
    Serial.print("LABEL ");
    Serial.println(CURRENT_LABEL);
  } else {
    Serial.print("UNKNOWN ");
    Serial.println(line);
  }
}

void setup() {
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);

  Serial.begin(115200);
  // Wait up to 5 s for USB serial — do NOT block forever or collect_data.py
  // will time out (common issue on Nano 33 BLE when the host opens the port).
  for (unsigned long start = millis(); !Serial && millis() - start < 5000UL;) {
    ;
  }

  if (!IMU.begin()) {
    Serial.println("ERR IMU init failed");
    while (1) {
      ;  // stop — cannot collect data without the IMU
    }
  }

  showLabelLed(CURRENT_LABEL);

  // Tell the host script the board is ready to receive "R ..." commands.
  Serial.println("READY");
  Serial.print("LABEL ");
  Serial.println(CURRENT_LABEL);
  Serial.print("RATE ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz (windows captured at 50 Hz)");
}

void loop() {
  if (recording) {
    return;  // do not process serial while a window is in progress
  }

  // Accumulate characters until newline, then handle one command.
  static char line[48];
  static uint8_t len = 0;

  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') {
      continue;
    }
    if (c == '\n') {
      line[len] = '\0';
      if (len > 0) {
        handleCommand(line);
      }
      len = 0;
    } else if (len < sizeof(line) - 1) {
      line[len++] = c;
    }
  }
}
