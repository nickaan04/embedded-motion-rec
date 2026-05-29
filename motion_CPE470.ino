// motion_test.ino
// Minimal test sketch — reads IMU, runs inference, prints result to Serial.
// No game logic, no LEDs. Just proves the model works on-device.
//
// Required Arduino libraries (install via Library Manager):
//   - Arduino_LSM9DS1
//   - Harvard_TinyMLx (Arduino_TensorFlowLite)

#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "motion_model.h"

// ---------------------------------------------------------------
// Model settings — must match the training notebook exactly
// ---------------------------------------------------------------
const int TIMESTEPS   = 50;
const int AXES        = 3;
const int NUM_CLASSES = 6;

const char* LABELS[NUM_CLASSES] = {
  "still",
  "shake",
  "tilt forward",
  "tilt backward",
  "tilt left",
  "tilt right"
};

// 20ms per sample = 50Hz
const int SAMPLE_INTERVAL_MS = 20;

// ---------------------------------------------------------------
// TFLite setup — Harvard_TinyMLx API requires an ErrorReporter
// ---------------------------------------------------------------
constexpr int kTensorArenaSize = 20 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor  = nullptr;
TfLiteTensor* output_tensor = nullptr;

// Op resolver — Flatten->Dense->Dense->Dense needs these 4 ops
tflite::MicroMutableOpResolver<2> resolver;

// ---------------------------------------------------------------
// IMU buffer
// ---------------------------------------------------------------
float imu_buffer[TIMESTEPS][AXES];

// ---------------------------------------------------------------
// Setup
// ---------------------------------------------------------------
void setup() {
  Serial.begin(9600);
  while (!Serial);

  Serial.println("=== Motion Recognition Test ===");

  // --- Init IMU ---
  if (!IMU.begin()) {
    Serial.println("ERROR: IMU init failed!");
    while (1);
  }
  Serial.print("IMU sample rate: ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");

  // --- Init TFLite ---
  tfl_model = tflite::GetModel(g_model);

  // Harvard_TinyMLx uses GetModel()->version() compared to a local constant
  // Skip the schema check or use the raw value (3)
  if (tfl_model->version() != 3) {
    Serial.println("WARNING: Unexpected model schema version");
  }

  // Register ops — Harvard_TinyMLx API
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  // resolver.AddReshape();
  // resolver.AddDequantize();

  // Harvard_TinyMLx MicroInterpreter requires ErrorReporter as 5th argument
  static tflite::MicroInterpreter static_interpreter(
      tfl_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors failed! Try increasing kTensorArenaSize.");
    while (1);
  }

  input_tensor  = interpreter->input(0);
  output_tensor = interpreter->output(0);

  // Print tensor details for verification
  Serial.print("Input shape: (");
  for (int i = 0; i < input_tensor->dims->size; i++) {
    Serial.print(input_tensor->dims->data[i]);
    if (i < input_tensor->dims->size - 1) Serial.print(", ");
  }
  Serial.println(")");
  Serial.print("Input type: ");
  Serial.println(input_tensor->type == kTfLiteInt8 ? "INT8" : "FLOAT32");

  Serial.println("Ready — press ENTER in Serial Monitor to classify.");
  Serial.println("----------------------------------------------");
}

// ---------------------------------------------------------------
// Collect exactly TIMESTEPS IMU samples into imu_buffer
// ---------------------------------------------------------------
void collectSamples() {
  Serial.println("Collecting samples...");
  for (int t = 0; t < TIMESTEPS; t++) {
    float ax, ay, az;

    unsigned long start = millis();
    while (!IMU.accelerationAvailable()) {
      if (millis() - start > 500) {
        Serial.println("WARNING: IMU timeout");
        break;
      }
    }

    IMU.readAcceleration(ax, ay, az);
    imu_buffer[t][0] = ax;
    imu_buffer[t][1] = ay;
    imu_buffer[t][2] = az;

    delay(SAMPLE_INTERVAL_MS);
  }
  Serial.println("Done collecting.");
}

// ---------------------------------------------------------------
// Run inference, return index of top class
// ---------------------------------------------------------------
int runInference() {
  float scale      = input_tensor->params.scale;
  int   zero_point = input_tensor->params.zero_point;

  int idx = 0;
  for (int t = 0; t < TIMESTEPS; t++) {
    for (int a = 0; a < AXES; a++) {
      float val = imu_buffer[t][a];
      int quantized = (int)roundf(val / scale) + zero_point;
      quantized = constrain(quantized, -128, 127);
      input_tensor->data.int8[idx++] = (int8_t)quantized;
    }
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Inference failed!");
    return -1;
  }

  // Find top class
  int    best_class = 0;
  int8_t best_score = output_tensor->data.int8[0];
  for (int i = 1; i < NUM_CLASSES; i++) {
    if (output_tensor->data.int8[i] > best_score) {
      best_score = output_tensor->data.int8[i];
      best_class = i;
    }
  }

  // Print all scores
  float out_scale      = output_tensor->params.scale;
  int   out_zero_point = output_tensor->params.zero_point;
  Serial.println("Scores:");
  for (int i = 0; i < NUM_CLASSES; i++) {
    float prob = (output_tensor->data.int8[i] - out_zero_point) * out_scale;
    Serial.print("  ");
    Serial.print(LABELS[i]);
    Serial.print(": ");
    Serial.println(prob, 3);
  }

  return best_class;
}

// ---------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------
void loop() {
  Serial.println("\nPress ENTER to start a classification...");

  while (!Serial.available());
  while (Serial.available()) Serial.read();

  collectSamples();

  int predicted = runInference();
  if (predicted >= 0) {
    Serial.print(">>> Predicted: ");
    Serial.println(LABELS[predicted]);
  }
}
