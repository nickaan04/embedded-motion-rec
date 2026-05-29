#ifndef MOTION_MODEL_H
#define MOTION_MODEL_H

// The trained motion classification model converted to a C array via xxd.
// Replace motion_model.cpp with the output of:
//   xxd -i model_int8.tflite > motion_model.cpp
// then fix the variable names to match these declarations.

extern const unsigned char g_model[];
extern const int g_model_len;

#endif  // MOTION_MODEL_H