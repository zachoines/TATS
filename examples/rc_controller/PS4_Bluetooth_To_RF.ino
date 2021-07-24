#include <PS4Controller.h>
#include <SPI.h>
#include <nRF24L01.h>
#include <RF24.h>
RF24 radio(14, 32); // CE & CS
const uint8_t pipeOut[] = { 0xE6, 0xE6, 0xE6, 0xE6, 0xE6 };

// Define data and formats
enum data { wLeft = 0, wRight = 1, sUp = 2, sDown = 3, t1 = 4, t2 = 5, t3 = 6, t4 = 7};
struct Signal {
  byte buffer[8];

  bool operator == (struct Signal& a)
  {
    for (int i = 0; i < 4; i++) {
      if (abs(buffer[i] - a.buffer[i]) > 0) {
        return false;
      }
    }
    
    return true;
  }

  struct Signal& operator = (struct Signal& a)
  {
    for (int i = 0; i < 8; i++) {
      this->buffer[i] = a.buffer[i];
    }

    return *this;
  }
};

// Define data types
union bytesToTimestamp {
  byte buffer[4];
  unsigned long timestamp;
} timestamp_converter;

// Define globals
Signal oldData;
Signal newData;

// Function hoisting here
int mapJoystickValues(int val, int lower, int middle, int upper, bool reverse);
unsigned long getTimestamp(struct Signal& s);

void setup() {
  Serial.begin(115200);
  PS4.begin("01:01:01:01:01:01");
  radio.begin();
  radio.openWritingPipe(pipeOut);
  radio.stopListening(); // Set to Transmitter mode
  if (!radio.isChipConnected ()) {
    Serial.println("There is an issue connecting to radio receiver.");
  }
  
  Serial.println("Starting radio transmitter...");  
}

void loop() {

  if (PS4.isConnected()) {
    oldData = newData;
    timestamp_converter.timestamp = millis();
    newData.buffer[data::wLeft] = mapJoystickValues(PS4.LStickY(), -128, 0, 127, true);
    newData.buffer[data::sUp] = 0;
    newData.buffer[data::wRight] = mapJoystickValues(PS4.RStickY(), -128, 0, 127, true);
    newData.buffer[data::sDown] = 0;
    newData.buffer[data::t1] = timestamp_converter.buffer[0];
    newData.buffer[data::t2] = timestamp_converter.buffer[1];
    newData.buffer[data::t3] = timestamp_converter.buffer[2];
    newData.buffer[data::t4] = timestamp_converter.buffer[3];
  
    if (!(newData == oldData)) {
      radio.write(&newData.buffer, 8);
    }  
  }
}

int mapJoystickValues(int val, int lower, int middle, int upper, bool reverse) {
  int clipped = constrain(val, lower, upper);
  int adjusted;  
  if (clipped < middle) {
    adjusted = map(clipped, lower, middle, 0, 127);
  } else {
    adjusted = map(clipped, middle, upper, 128, 255);
  }
    
  return (reverse ? 255 - adjusted : adjusted);
}

unsigned long getTimestamp(struct Signal& s) {

  timestamp_converter.buffer[0] = s.buffer[data::t1];
  timestamp_converter.buffer[1] = s.buffer[data::t2];
  timestamp_converter.buffer[2] = s.buffer[data::t3];
  timestamp_converter.buffer[3] = s.buffer[data::t4];

  return timestamp_converter.timestamp;
}