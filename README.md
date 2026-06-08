# CPE 470 Final Project

### Nickaan Jahadi and Michael Drobeck  

This project creates an embedded ML motion-recognition game on the Arduino Nano 33 BLE Sense using its onboard IMU. The system randomly prompts the user to perform one of six motions: shaking the board, tilting it left, right, forward, or backward, or holding it still, each shown by a unique LED color. After the prompt, the user performs the motion while the board records a 1-second accelerometer window, signified by a white LED flash from the board. Then, an on-device INT8 TFLite model trained on 600 labeled samples classifies the gesture and checks it against the prompt using a confidence threshold. Beyond single-motion rounds in a 10-round scoring game, it also includes a Bop It/Simon Says mode where players must repeat growing sequences of motions, increasing difficulty and making the game more engaging. Game selection and feedback are provided over the Serial monitor, and the game concludes upon any incorrect gestures.

