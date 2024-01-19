/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <TensorFlowLite.h>

#include "main_functions.h"

#include "accelerometer_handler.h"
#include "constants.h"
#include "gesture_predictor.h"
#include "magic_wand_model_data.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// #include "Freenove_WS2812_Lib_for_ESP32.h"
// #define LED_PIN 1
// Freenove_ESP32_WS2812 led = Freenove_ESP32_WS2812(1, LED_PIN, 0, TYPE_GRB);
// void setLED(uint8_t r,uint8_t g,uint8_t b) {
//   led.setLedColorData(0, r, g, b);
//   led.show();
// }

// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
int input_length;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 94020;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

float movement_threshold;

// The name of this function is important for Arduino compatibility.
void setup() {
  Serial.begin(115200);
  while(!Serial);

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  static tflite::AllOpsResolver resolver;
  // static tflite::MicroMutableOpResolver<10> micro_mutable_op_resolver;  // NOLINT

  // micro_mutable_op_resolver.AddDepthwiseConv2D();
  // micro_mutable_op_resolver.AddFullyConnected();
  // micro_mutable_op_resolver.AddConv2D();
  // micro_mutable_op_resolver.AddMaxPool2D();
  // micro_mutable_op_resolver.AddSoftmax();
  // micro_mutable_op_resolver.AddMul();
  // micro_mutable_op_resolver.AddAdd();
  // micro_mutable_op_resolver.AddMean();
  // micro_mutable_op_resolver.AddExpandDims();
  // micro_mutable_op_resolver.AddBuiltin(
  //     tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
  //     tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  // micro_mutable_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
  //                              tflite::ops::micro::Register_MAX_POOL_2D());
  // micro_mutable_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
  //                              tflite::ops::micro::Register_CONV_2D());
  // micro_mutable_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
  //                              tflite::ops::micro::Register_FULLY_CONNECTED());
  // micro_mutable_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
  //                              tflite::ops::micro::Register_SOFTMAX());
  // micro_mutable_op_resolver.AddBuiltin(tflite::BuiltinOperator_MUL,
  //                              tflite::ops::micro::Register_MUL());
  // micro_mutable_op_resolver.AddBuiltin(tflite::BuiltinOperator_ADD,
  //                              tflite::ops::micro::Register_ADD());
  // micro_mutable_op_resolver.AddBuiltin(tflite::BuiltinOperator_MEAN,
  //                              tflite::ops::micro::Register_MEAN());

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input tensor.
  model_input = interpreter->input(0);
  // if ((model_input->dims->size != 3) || (model_input->dims->data[0] != 1) ||  // 3 1
  //     (model_input->dims->data[1] != 128) ||                                  // 384
  //     (model_input->dims->data[2] != kChannelNumber) ||                       // 9
  //     (model_input->type != kTfLiteFloat32)) {                                // 1
  //   Serial.println(model_input->dims->size);
  //   Serial.println(model_input->dims->data[0]);
  //   Serial.println(model_input->dims->data[1]);
  //   Serial.println(model_input->dims->data[2]);
  //   Serial.println(model_input->type);
  //   Serial.println(kTfLiteFloat32);
  //   Serial.println(kChannelNumber);
  //   TF_LITE_REPORT_ERROR(error_reporter,
  //                        "Bad input tensor parameters in model :(");
  //   return;
  // }

  input_length = model_input->bytes / sizeof(float);

  TfLiteStatus setup_status = SetupAccelerometer();
  movement_threshold = CalibrateAccelerometer();
  if (setup_status != kTfLiteOk) {
    Serial.println("Set up failed\n");
  }
}

void print_prediction_data() {
  const float* input_data = model_input->data.f;
  for(int i = 0; i<input_length; i=i+3) {
    // Serial.print("x:");
    // Serial.print(input_data[i]);
    // Serial.print(",y:");
    // Serial.print(input_data[i+1]);
    // Serial.print(",z:");
    Serial.println(input_data[i+2]);
  }
}

bool IsMoving() {
  // Look at the most recent accelerometer values.
  const float* input_data = model_input->data.f;
  const float last_x = input_data[input_length - 3];
  const float last_y = input_data[input_length - 2];
  const float last_z = input_data[input_length - 1];

  // Figure out the total amount of acceleration being felt by the device.
  // const float last_x_squared = last_x * last_x;
  // const float last_y_squared = last_y * last_y;
  // const float last_z_squared = last_z * last_z;
  const float acceleration_magnitude =
      last_x + last_y + last_z;
  // Acceleration is in milli-Gs, so normal gravity is 1,000 units.
  const float gravity = 0.0f;

  // Subtract out gravity to get the actual movement magnitude.
  const float movement = acceleration_magnitude - gravity;

  // How much acceleration is needed before it's considered movement.
  const bool is_moving = (movement > (movement_threshold + 0.5)) || (movement < (movement_threshold - 0.5));

  // Serial.println(is_moving);
  return is_moving;
}

// This is the regular function we run to recognize gestures from a pretrained
// model.
void RecognizeGestures() {
  const bool is_moving = IsMoving();

  // Static state used to control the capturing process.
  static int counter = 0;
  static enum {
    ePendingStillness,
    eInStillness,
    ePendingMovement,
    eRecordingGesture
  } state = ePendingStillness;
  static int still_found_time;
  static int gesture_start_time;
  // State machine that controls gathering user input.

  switch (state) {
    case ePendingStillness: {
      if (!is_moving) {
        still_found_time = counter;
        state = eInStillness;
      }
    } break;

    case eInStillness: {
      if (is_moving) {
        state = ePendingStillness;
      } else {
        const int duration = counter - still_found_time;
        if (duration > 3) {
          state = ePendingMovement;
        }
      }
    } break;

    case ePendingMovement: {
      if (is_moving) {
        state = eRecordingGesture;
        gesture_start_time = counter;
      }
    } break;

    case eRecordingGesture: {
      const int recording_time = 256; // half of 512 so step is in middle of data
      // wait until enough data ready
      if ((counter - gesture_start_time) > recording_time) {
        // Run inference, and report any error.
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
          Serial.printf("Invoke failed on index: %d\n",
                               begin_index);
          return;
        }
        // print_prediction_data();
        const float* prediction_scores = interpreter->output(0)->data.f;
        const int found_gesture = PredictGesture(prediction_scores);

        // Serial.println(found_gesture);
        // Produce an output
        HandleOutput(found_gesture);

        delay(100);

        movement_threshold = CalibrateAccelerometer();

        state = ePendingStillness;
      }
    } break;

    default: {
      Serial.println("Logic error - unknown state");
    } break;
  }

  // Increment the timing counter.
  ++counter;
}

// If you need to gather training data, call this function from the main loop
// and it will guide the user through contributing data.
// The output that's logged to the console can be fed into the Python training
// scripts for this example.
void CaptureGestureData() {
  const bool is_moving = IsMoving();

  // Static state used to control the capturing process.
  static int counter = 0;
  static int gesture_count = 0;
  static enum {
    eStarting,
    ePendingStillness,
    eInStillness,
    ePendingMovement,
    eRecordingGesture
  } state = eStarting;
  static int still_found_time;
  static int gesture_start_time;
  static const char* next_gesture = nullptr;
  // State machine that controls gathering user input.
  switch (state) {
    case eStarting: {
      if (!next_gesture || (strcmp(next_gesture, "nostep") == 0)) {
        next_gesture = "step";
      } else if (strcmp(next_gesture, "nostep") == 0) {
        next_gesture = "nostep";
      }
      Serial.println("# Hold the wand still");
      state = ePendingStillness;
    } break;

    case ePendingStillness: {
      if (!is_moving) {
        still_found_time = counter;
        state = eInStillness;
      }
    } break;

    case eInStillness: {
      if (is_moving) {
        state = ePendingStillness;
      } else {
        const int duration = counter - still_found_time;
        if (duration > 25) {
          state = ePendingMovement;
          Serial.printf("# When you're ready, perform the %s gesture",
                               next_gesture);
        }
      }
    } break;

    case ePendingMovement: {
      if (is_moving) {
        state = eRecordingGesture;
        gesture_start_time = counter;
        Serial.printf("# Perform the %s gesture now",
                             next_gesture);
      }
    } break;

    case eRecordingGesture: {
      const int recording_time = 100;
      if ((counter - gesture_start_time) > recording_time) {
        ++gesture_count;
        Serial.println("****************");
        Serial.printf("gesture: %s", next_gesture);
        const float* input_data = model_input->data.f;
        for (int offset = recording_time - 10; offset > 0; --offset) {
          const int array_offset = (input_length - (offset * 3));
          const int x = static_cast<int>(input_data[array_offset + 0]);
          const int y = static_cast<int>(input_data[array_offset + 1]);
          const int z = static_cast<int>(input_data[array_offset + 2]);
          Serial.printf("x: %d y:%d z:%d", x, y, z);
        }
        Serial.println("~~~~~~~~~~~~~~~~");
        Serial.printf("# %d gestures recorded",
                             gesture_count);
        state = eStarting;
      }
    } break;

    default: {
      Serial.println("Logic error - unknown state");
    } break;
  }

  // Increment the timing counter.
  ++counter;
}

void loop() {
  unsigned long startTime = millis(); // Get the current time in milliseconds
  delay(5);
 
  // Attempt to read new data from the accelerometer.
  bool got_data =
      ReadAccelerometer(model_input->data.f, input_length);
  // If there was no new data, wait until next time.
  if (!got_data) return;

  // In the future we should decide whether to capture data based on a user
  // action (like pressing a button), but since some of the devices we're
  // targeting don't have any built-in input devices you'll need to manually
  // switch between recognizing gestures and capturing training data by changing
  // this variable and recompiling.
  const bool should_capture_data = false;
  if (should_capture_data) {
    CaptureGestureData();
  } else {
    RecognizeGestures();
    // Serial.println(1000.0/(millis()-startTime));
  }
}
