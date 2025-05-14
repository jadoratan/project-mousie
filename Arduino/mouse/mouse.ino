#include <Wire.h>
#include <Mouse.h>
#include <Adafruit_LSM9DS1.h>
#include <Adafruit_Sensor.h>

Adafruit_LSM9DS1 lsm = Adafruit_LSM9DS1();

const float sensitivity = 8.0;   // Mouse movement multiplier
const float deadzone = 0.5;      // Ignore small jitter

void setup() {
  Serial.begin(115200);
  while (!Serial);  // Wait for Serial Monitor to open

  Wire.begin();     // I2C (SDA = A4, SCL = A5)
  delay(500);

  if (!lsm.begin()) {
    Serial.println("❌ LSM9DS1 not detected. Check wiring and CS_AG/CS_M!");
    while (true) {
      digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
      delay(250);
    }
  }

  lsm.setupGyro(lsm.LSM9DS1_GYROSCALE_245DPS);

  Mouse.begin();
  Serial.println("✅ Sensor initialized. Mouse control active.");
}

void loop() {
  sensors_event_t accel, mag, gyro, temp;
  lsm.getEvent(&accel, &mag, &gyro, &temp);

  float gx = gyro.gyro.x;
  float gy = gyro.gyro.y;

  // Deadzone filter
  if (abs(gx) < deadzone) gx = 0;
  if (abs(gy) < deadzone) gy = 0;

  int dx = int(gx * sensitivity);
  int dy = int(gy * sensitivity);

  // Clamp movement
  dx = constrain(dx, -20, 20);
  dy = constrain(dy, -20, 20);

  // Move the mouse
  Mouse.move(dx, dy);

  // Print to Serial Monitor
  Serial.print("Gyro X: "); Serial.print(gx, 2);
  Serial.print(" | Y: "); Serial.print(gy, 2);
  Serial.print(" → dx: "); Serial.print(dx);
  Serial.print(" | dy: "); Serial.println(dy);

  delay(10);  // Adjust for responsiveness
}
