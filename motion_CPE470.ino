#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "motion_model.h"
#include "motion_model_settings.h"

const int TIMESTEPS = kTimesteps;
const int AXES = kAxes;
const int NUM_CLASSES = kNumClasses;
const int SAMPLE_INTERVAL_MS = 20;

// Classes for diff types of motion
#define STILL         0
#define SHAKE         1
#define TILT_FORWARD  2
#define TILT_BACKWARD 3
#define TILT_LEFT     4
#define TILT_RIGHT    5

const char* LABELS[NUM_CLASSES] = {
  "still",
  "shake",
  "tilt forward",
  "tilt backward",
  "tilt left",
  "tilt right"
};

constexpr int kTensorArenaSize = 20 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

tflite::MicroMutableOpResolver<2> resolver;

float imu_buffer[TIMESTEPS][AXES];

const float CONFIDENCE_THRESHOLD = 0.6;

const int TOTAL_ROUNDS = 10; //num rounds for motion game (not simon says)
int score = 0;

const int MAX_SEQUENCE = 20; //max rounds for simon says
int sequence[MAX_SEQUENCE];
int sequence_length = 0;

void setLED(int r, int b, int g) {
  digitalWrite(LEDR, r ? LOW : HIGH);
  digitalWrite(LEDG, g ? LOW : HIGH);
  digitalWrite(LEDB, b ? LOW : HIGH);
}

void ledsOff() {
  setLED(0, 0, 0);
}

void showMotionColor(int motionClass) { //diff LED configs for each motion class
  switch (motionClass) {
    case STILL:         setLED(0, 1, 1); break;
    case SHAKE:         setLED(1, 1, 0); break;
    case TILT_FORWARD:  setLED(0, 1, 0); break;
    case TILT_BACKWARD: setLED(1, 0, 1); break;
    case TILT_LEFT:     setLED(0, 0, 1); break;
    case TILT_RIGHT:    setLED(1, 0, 0); break;
    default:                  ledsOff(); break;
  }
}

void collectSamples() { // Record 50 samples of xyz acceleration from IMU
  for (int t = 0; t < TIMESTEPS; t++) {
    float ax, ay, az;
    unsigned long start = millis();

    while (!IMU.accelerationAvailable()) { //wait for new reading
      if (millis() - start > 500) break;
    }

    IMU.readAcceleration(ax, ay, az);

    imu_buffer[t][0] = ax;
    imu_buffer[t][1] = ay;
    imu_buffer[t][2] = az;

    delay(SAMPLE_INTERVAL_MS);
  }
}

//sends IMU buffer to model and returns prediction
int runInference(float* confidence) {
  float scale = input_tensor->params.scale;
  int zero_point = input_tensor->params.zero_point;

  int idx = 0;
  //quantize float IMU values into int8 values expected by the model
  for (int t = 0; t < TIMESTEPS; t++) {
    for (int a = 0; a < AXES; a++) {
      int q = (int)roundf(imu_buffer[t][a] / scale) + zero_point;
      q = constrain(q, -128, 127);
      input_tensor->data.int8[idx++] = (int8_t)q;
    }
  }

  if (interpreter->Invoke() != kTfLiteOk) { //run the model
    return -1;
  }

  float out_scale = output_tensor->params.scale;
  int out_zero_point = output_tensor->params.zero_point;

  int best_class = 0;
  float best_prob = 0.0;

  for (int i = 0; i < NUM_CLASSES; i++) {
    float prob = (output_tensor->data.int8[i] - out_zero_point) * out_scale;

    if (prob > best_prob) {
      best_prob = prob;
      best_class = i;
    }
  }

  *confidence = best_prob;
  return best_class;
}

void waitForEnter() { //wait for user to press Enter on serial before moving forward
  while (!Serial.available());
  while (Serial.available()) Serial.read();
}

char chooseGame() { //Wait for user to choose game
  Serial.println();
  Serial.println("Choose a game:");
  Serial.println("1 = Motion Prompt Game");
  Serial.println("2 = Simon Says Motion Game");
  Serial.println("Enter 1 or 2:");

  while (true) { //get serial reading for game - only accept '1' or '2', everything else is invalid
    if (Serial.available()) {
      char choice = Serial.read();

      while (Serial.available()) Serial.read();

      if (choice == '1' || choice == '2') {
        return choice;
      }

      Serial.println("Invalid choice. Enter 1 or 2:");
    }
  }
}

void playPromptRound(int round) { //play motion round - not simon says
  int target = random(NUM_CLASSES); //figure out random motion

  Serial.println();
  Serial.print("Round: ");
  Serial.print(round);
  Serial.print(" / ");
  Serial.println(TOTAL_ROUNDS);

  Serial.print("Perform motion shown by LED color: ");
  // Serial.println(LABELS[target]); //debugging colors

  showMotionColor(target);
  delay(1000); //change for how long you want light to show
  ledsOff();
  delay(300);
  setLED(1, 1, 1); //let user know we are collecting

  collectSamples();
  ledsOff();

  float confidence = 0.0;
  int predicted = runInference(&confidence); //figure out which motion

  Serial.print("Predicted: ");
  Serial.println(predicted >= 0 ? LABELS[predicted] : "ERROR");

  Serial.print("Confidence: ");
  Serial.println(confidence, 3);

  //player scores only if confidence is greater than threshold and if prediction is what was given to user
  if (predicted == target && confidence >= CONFIDENCE_THRESHOLD) {
    score++;
    Serial.println("Correct!");
  } else {
    Serial.print("Wrong! Expected: ");
    Serial.println(LABELS[target]);
  }

  Serial.print("Score: ");
  Serial.print(score);
  Serial.print(" / ");
  Serial.println(round);

  delay(1500);
}

void playPromptGame() { //full motion game (just printing stuff mostly)
  score = 0;

  Serial.println();
  Serial.println("----- Motion Prompt Game -----");
  Serial.println("Match the LED prompt with the correct motion.");
  delay(1000);

  for (int round = 1; round <= TOTAL_ROUNDS; round++) {
    playPromptRound(round);
  }

  Serial.println();
  Serial.println("=== GAME OVER ===");

  Serial.print("Final score: ");
  Serial.print(score);
  Serial.print(" / ");
  Serial.println(TOTAL_ROUNDS);

  if (score == TOTAL_ROUNDS) {
    Serial.println("Perfect score!");
  } else if (score >= TOTAL_ROUNDS / 2) {
    Serial.println("Good job!");
  } else {
    Serial.println("Bad job.");
  }

  ledsOff();
}

void playSequence() { //play current simon says sequence
  int flash_ms = max(300, 600 - ((sequence_length / 5) * 100)); //delay gets faster over time

  Serial.println("Watch the sequence:");

  for (int i = 0; i < sequence_length; i++) {
    Serial.print("  ");
    // Serial.println(LABELS[sequence[i]]); //debugging stuff for LEDS matching with moves

    showMotionColor(sequence[i]);
    delay(flash_ms);
    ledsOff();
    delay(300);
  }
}

int getPlayerMove(int step) { //get one motion from player during simon says sequence
  Serial.print("Move ");
  Serial.print(step + 1);
  Serial.print(": ");
  setLED(1,1,1); delay(50); //white LED lets player know its ready for collecting

  collectSamples();
  ledsOff(); delay(250); //turn off after recording

  float confidence = 0.0;
  int predicted = runInference(&confidence);

  Serial.print(predicted >= 0 ? LABELS[predicted] : "ERROR");
  Serial.print(" (");
  Serial.print(confidence, 2);
  Serial.println(")");

  if (confidence < CONFIDENCE_THRESHOLD) {
    return -1;
  }

  return predicted;
}

void playSimonGame() { //play actual game
  sequence_length = 0;
  bool game_over = false;

  Serial.println();
  Serial.println("----- Simon Says Motion Game -----");
  Serial.println("Repeat the full motion sequence each round.");
  delay(1000);

  while (!game_over && sequence_length < MAX_SEQUENCE) {
    sequence[sequence_length] = random(NUM_CLASSES); //add a random motion every round
    sequence_length++;

    Serial.println();
    Serial.print("----- Round ");
    Serial.print(sequence_length);
    Serial.println(" -----");

    delay(500);
    playSequence();

    delay(800);
    Serial.println("Your turn");

    for (int step = 0; step < sequence_length; step++) {
      int player_move = getPlayerMove(step);

      if (player_move != sequence[step]) {
        Serial.println("Wrong");

        Serial.print("Expected: ");
        Serial.println(LABELS[sequence[step]]);

        Serial.print("Got: ");
        Serial.println(player_move >= 0 ? LABELS[player_move] : "unclear");

        game_over = true;
        break;
      } else {
        delay(200); //next round
      }
    }

    if (!game_over) {
      Serial.println("Sequence complete --- Next round."); //print blank lines so user can't look at serial monitor and get sequence during game
      delay(1000);
      for (int i = 0; i < 40; i++) {
        Serial.println();
      }
    }
  }

  Serial.println();
  Serial.println("=== GAME OVER ===");

  Serial.print("Your highest round: ");
  Serial.println(sequence_length - 1);

  if (sequence_length - 1 == 0) {
    Serial.println("Do better.");
  } else if (sequence_length - 1 < 5) {
    Serial.println("Solid I guess.");
  } else {
    Serial.println("Nice job.");
  }

  ledsOff();
}

void setup() {
  Serial.begin(9600);
  while (!Serial);

  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);

  ledsOff();

  setLED(1, 0, 0); delay(200);
  setLED(0, 1, 0); delay(200);
  setLED(0, 0, 1); delay(200);
  ledsOff();

  Serial.println("=== Motion Game Menu ===");

  if (!IMU.begin()) { //start IMU sensor
    Serial.println("IMU init failed!");
    while (1);
  }

  tfl_model = tflite::GetModel(g_model); //load model

  if (tfl_model->version() != 3) {
    Serial.println("WARNING: Unexpected model schema version");
  }

  //only 2 operations exist in model
  resolver.AddFullyConnected();
  resolver.AddSoftmax();

  static tflite::MicroInterpreter static_interpreter(
    tfl_model,
    resolver,
    tensor_arena,
    kTensorArenaSize,
    error_reporter
  );

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors failed!");
    while (1);
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  randomSeed(analogRead(0));

  Serial.println("Model loaded.");
}

void loop() {
  char choice = chooseGame();

  if (choice == '1') {
    playPromptGame();
  } else if (choice == '2') {
    playSimonGame();
  }

  Serial.println();
  Serial.println("Press ENTER to return to the game menu...");
  waitForEnter();
}
