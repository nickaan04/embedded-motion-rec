# CPE 470 Final Project

### Nickaan Jahadi and Michael Drobeck  

This project aims to create an embedded ML motion-recognition game using the Arduino Nano 33 BLE Sense’s onboard IMU. The system will randomly prompt a user to perform a specific physical motion, such as shaking the board, tilting it left, tilting it right, tilting it forward, tilting it backward, or holding it still. Each motion will be represented by a unique LED color, allowing the board to communicate the requested action without needing any external hardware. After the prompt, the user performs the motion, and an on-device ML model classifies the accelerometer data to determine whether the correct action was completed. If single-motion classification works reliably, the system can be extended to recognize short sequences of motions, increasing the difficulty and making the game more engaging.

