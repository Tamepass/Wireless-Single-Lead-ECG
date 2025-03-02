#include <WiFi.h>  // WiFi communication

// Define ECG sensor pins
#define ECG_PIN 34   // AD8232 Output (Analog)
#define LO_PLUS 35   // Lead-off detection (+)
#define LO_MINUS 32  // Lead-off detection (-)

// WiFi Credentials (Modify these)
const char* ssid = "Your_SSID";
const char* password = "Your_PASSWORD";

// Webserver setup
WiFiServer server(80);

void setup() {
    Serial.begin(115200);   // Serial Monitor
    
    // WiFi Setup
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
    }
    server.begin(); // Start server
    
    // Configure AD8232 input pins
    pinMode(ECG_PIN, INPUT);
    pinMode(LO_PLUS, INPUT);
    pinMode(LO_MINUS, INPUT);
}

void loop() {
    int ecgValue = analogRead(ECG_PIN); // Read ECG data

    // Send data over WiFi
    WiFiClient client = server.available();
    if (client) {
        client.print("ECG: ");
        client.println(ecgValue);
        client.stop();
    }

    delay(10); // Sampling rate control
}
