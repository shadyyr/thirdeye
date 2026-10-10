// Third Eye four-zone ESP32 haptic controller.
// GPIO pins control MOSFET gates; motors need their own suitable power path.

const uint32_t SERIAL_BAUD_RATE = 115200;
const unsigned long COMMAND_TIMEOUT_MS = 600;
const uint32_t PWM_FREQUENCY_HZ = 200;
const uint8_t PWM_RESOLUTION_BITS = 8;

// Update these four assignments to match the belt wiring.
const uint8_t FRONT_PIN = 25;
const uint8_t RIGHT_PIN = 26;
const uint8_t REAR_PIN = 27;
const uint8_t LEFT_PIN = 14;

const uint8_t FRONT_CHANNEL = 0;
const uint8_t RIGHT_CHANNEL = 1;
const uint8_t REAR_CHANNEL = 2;
const uint8_t LEFT_CHANNEL = 3;

String serialBuffer;
unsigned long lastValidCommandAt = 0;

void setupPwm(uint8_t pin, uint8_t channel) {
  ledcSetup(channel, PWM_FREQUENCY_HZ, PWM_RESOLUTION_BITS);
  ledcAttachPin(pin, channel);
  ledcWrite(channel, 0);
}

void stopAllMotors() {
  ledcWrite(FRONT_CHANNEL, 0);
  ledcWrite(RIGHT_CHANNEL, 0);
  ledcWrite(REAR_CHANNEL, 0);
  ledcWrite(LEFT_CHANNEL, 0);
}

void activateZone(const String& zone, int intensity) {
  stopAllMotors();
  int duty = map(constrain(intensity, 0, 100), 0, 100, 0, 255);
  if (zone == "FRONT") {
    ledcWrite(FRONT_CHANNEL, duty);
  } else if (zone == "RIGHT") {
    ledcWrite(RIGHT_CHANNEL, duty);
  } else if (zone == "REAR") {
    ledcWrite(REAR_CHANNEL, duty);
  } else if (zone == "LEFT") {
    ledcWrite(LEFT_CHANNEL, duty);
  }
}

bool parseAndApplyCommand(String line) {
  line.trim();
  int separator = line.indexOf(',');
  if (separator <= 0 || separator != line.lastIndexOf(',')) {
    Serial.println("ERR malformed command");
    return false;
  }

  String zone = line.substring(0, separator);
  String intensityText = line.substring(separator + 1);
  zone.trim();
  intensityText.trim();
  zone.toUpperCase();

  for (size_t i = 0; i < intensityText.length(); i++) {
    if (!isDigit(intensityText[i])) {
      Serial.println("ERR intensity must be 0-100");
      return false;
    }
  }
  int intensity = intensityText.toInt();
  if (intensity < 0 || intensity > 100) {
    Serial.println("ERR intensity out of range");
    return false;
  }
  if (zone == "STOP") {
    if (intensity != 0) {
      Serial.println("ERR STOP requires zero intensity");
      return false;
    }
    stopAllMotors();
  } else if (zone == "FRONT" || zone == "RIGHT" || zone == "REAR" || zone == "LEFT") {
    activateZone(zone, intensity);
  } else {
    Serial.println("ERR unknown zone");
    return false;
  }

  lastValidCommandAt = millis();
  Serial.print("OK zone=");
  Serial.print(zone);
  Serial.print(" intensity=");
  Serial.println(intensity);
  return true;
}

void setup() {
  Serial.begin(SERIAL_BAUD_RATE);
  setupPwm(FRONT_PIN, FRONT_CHANNEL);
  setupPwm(RIGHT_PIN, RIGHT_CHANNEL);
  setupPwm(REAR_PIN, REAR_CHANNEL);
  setupPwm(LEFT_PIN, LEFT_CHANNEL);
  stopAllMotors();
  lastValidCommandAt = millis();
  Serial.println("Third Eye haptic controller ready");
}

void loop() {
  while (Serial.available() > 0) {
    char incoming = static_cast<char>(Serial.read());
    if (incoming == '\n') {
      parseAndApplyCommand(serialBuffer);
      serialBuffer = "";
    } else if (incoming != '\r' && serialBuffer.length() < 40) {
      serialBuffer += incoming;
    } else if (serialBuffer.length() >= 40) {
      serialBuffer = "";
      Serial.println("ERR command too long");
    }
  }

  if (millis() - lastValidCommandAt > COMMAND_TIMEOUT_MS) {
    stopAllMotors();
  }
}
