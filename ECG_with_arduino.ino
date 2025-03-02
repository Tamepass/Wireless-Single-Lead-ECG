#include <SoftwareSerial.h>

SoftwareSerial BT(2, 3); // (HC-05 TX to Arduino Pin 2, HC-05 RX to Pin 3)

void setup() {
  Serial.begin(115200);  // For Serial Monitor
  Serial.begin(9600); // Communicate with Virtual Terminal     
  pinMode(A0, INPUT);
  pinMode(8, INPUT);
  pinMode(9, INPUT);
}

void loop() {
  // int ecgValue = analogRead(A0); // Read ECG Signal
  // Serial.println(ecgValue);      // Print to Serial Monitor
  // BT.println(ecgValue);          // Send to Bluetooth Device (Mobile App)
  // delay(10);
   int ecgValue = analogRead(A0);
    Serial.println(ecgValue); // Send ECG data to Virtual Terminal
    delay(10);
}

