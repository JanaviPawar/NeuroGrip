// ============================================================
//  NeuroGrip ESP32 firmware  v3.0
//
//  What changed from v2.5
//   1. The MPU is read 100 times a second, the
//      slow part (gravity, steady tilt) is removed, and the leftover
//      shaking is turned into a 0-1 "tremor level" once every 50 ms.
//      If the MPU6050 isn't found at boot, it falls back to the GPIO32 pot,
//      so the board still works exactly like before.
//   
//   3. Compiles on both ESP32 Arduino core 2.x and 3.x (the timer API
//      changed in 3.x, which is what a fresh Arduino IDE installs).
//   4. No library needed for the MPU6050 (plain Wire register reads).
//
//  Serial protocol is UNCHANGED:  ESP32 -> PC  "fsr gsr tremor\n" (0-1 floats)
//                                 PC -> ESP32  "ALERT:n"  and  "DISP:State|conf|score|gci"
//  At boot it also prints "MPU_OK" or "MPU_MISSING_USING_POT".
// ============================================================

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <esp_arduino_version.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/semphr.h>

// ---------------- pins ----------------
#define FSR_PIN     34
#define GSR_PIN     35
#define TREMOR_PIN  32      // fallback pot, used only if the MPU6050 isn't found
#define BUZZER_PIN  25
#define LED_PIN      2

// ---------------- OLED ----------------
#define OLED_W      128
#define OLED_H       64
#define OLED_RESET   -1
#define OLED_ADDR  0x3C
Adafruit_SSD1306 display(OLED_W, OLED_H, &Wire, OLED_RESET);
bool oled_ok = false;

// ---------------- MPU6050 ----------------
#define MPU_ADDR        0x68     // 0x69 if the AD0 pin is tied to 3.3V
#define MPU_RATE_HZ     100
// Full-scale values: the shaking level that maps to tremor = 1.0.
// Tune these with collect_data.py's sensor check: holding the wheel still
// should read roughly 0.00-0.10, deliberate trembling 0.4-0.9.
#define TREMOR_ACC_FS_G     0.50f    // vibration in g
#define TREMOR_GYRO_FS_DPS  240.0f    // rotation jitter in degrees/second
bool mpu_ok = false;

// ---------------- sampling ----------------
#define SAMPLE_RATE_HZ  20
#define EMA_ALPHA       0.35f

// ---------------- shared state ----------------
volatile int alert_level   = 0;      // 0 SAFE, 1 CAUTION, 2 WARNING, 3 CRITICAL
String       state_label   = "Relax";
float        confidence    = 0.0f;
float        fatigue_score = 0.0f;
float        gci_val       = 1.0f;

typedef struct {
  float fsr_norm, gsr_norm, tremor_norm;
  float stress_score;
} ProcessedSample_t;

ProcessedSample_t latest_proc = {0};
SemaphoreHandle_t latest_proc_mutex;
SemaphoreHandle_t stress_mutex;
SemaphoreHandle_t data_ready_sem;
SemaphoreHandle_t i2c_mutex;         // NEW: OLED + MPU share the bus
SemaphoreHandle_t tremor_mutex;      // NEW: protects the tremor accumulators
hw_timer_t*       sample_timer = NULL;

// tremor accumulators (filled by mpuTask, emptied by sensorTask every 50 ms)
double trem_acc_sq = 0, trem_gyro_sq = 0;
uint32_t trem_count = 0;
float last_tremor = 0.0f;

// ============================================================
//  TIMER ISR (20 Hz)
// ============================================================
void IRAM_ATTR onSampleTimer() {
  BaseType_t xHigher = pdFALSE;
  xSemaphoreGiveFromISR(data_ready_sem, &xHigher);
  if (xHigher) portYIELD_FROM_ISR();
}

// ============================================================
//  ADC READ (oversampled, 0-1)
// ============================================================
inline float read_normalized(int pin) {
  uint32_t sum = 0;
  for (int i = 0; i < 4; i++) {
    sum += analogRead(pin);
    delayMicroseconds(30);
  }
  return (sum / 4) / 4095.0f;
}

float ema_fsr = 0, ema_gsr = 0, ema_trem = 0;
inline float ema_filter(float prev, float nv) {
  return EMA_ALPHA * nv + (1.0f - EMA_ALPHA) * prev;
}

// ============================================================
//  MPU6050 low-level helpers (call with i2c_mutex held)
// ============================================================
bool mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

bool mpuInit() {
  Wire.beginTransmission(MPU_ADDR);
  if (Wire.endTransmission() != 0) return false;        // nothing answered at 0x68
  bool ok = true;
  ok &= mpuWrite(0x6B, 0x00);   // wake up, internal clock
  delay(50);
  ok &= mpuWrite(0x19, 0x09);   // sample rate = 1 kHz / (1+9) = 100 Hz
  ok &= mpuWrite(0x1A, 0x03);   // digital low-pass ~44 Hz (keeps the 3-12 Hz tremor band)
  ok &= mpuWrite(0x1B, 0x08);   // gyro  +/-500 dps  (65.5 LSB per dps)
  ok &= mpuWrite(0x1C, 0x00);   // accel +/-2 g      (16384 LSB per g)
  return ok;
}

bool mpuRead(float &ax, float &ay, float &az, float &gx, float &gy, float &gz) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)14) != 14) return false;
  int16_t raw[7];
  for (int i = 0; i < 7; i++) {
    uint8_t hi = Wire.read();            // read high byte first, then low
    uint8_t lo = Wire.read();            // (two separate statements: C++ doesn't fix the order inside one expression)
    raw[i] = (int16_t)((hi << 8) | lo);
  }
  ax = raw[0] / 16384.0f; ay = raw[1] / 16384.0f; az = raw[2] / 16384.0f;
  // raw[3] is temperature, unused
  gx = raw[4] / 65.5f;    gy = raw[5] / 65.5f;    gz = raw[6] / 65.5f;
  return true;
}

// ============================================================
//  TASK: MPU (Core 1, priority 3) — 100 Hz
//  Removes gravity / slow tilt with a slow average, keeps the shaking.
// ============================================================
void mpuTask(void* param) {
  float g_avg = 1.0f;                      // slow average of |acceleration| (gravity)
  float gx_avg = 0, gy_avg = 0, gz_avg = 0; // slow average of rotation (gyro offset / steering)
  bool first = true;
  TickType_t last = xTaskGetTickCount();
  for (;;) {
    vTaskDelayUntil(&last, pdMS_TO_TICKS(1000 / MPU_RATE_HZ));
    float ax, ay, az, gx, gy, gz;
    bool ok;
    xSemaphoreTake(i2c_mutex, portMAX_DELAY);
    ok = mpuRead(ax, ay, az, gx, gy, gz);
    xSemaphoreGive(i2c_mutex);
    if (!ok) continue;

    float amag = sqrtf(ax * ax + ay * ay + az * az);
    if (first) { g_avg = amag; gx_avg = gx; gy_avg = gy; gz_avg = gz; first = false; }
    g_avg  = 0.90f * g_avg  + 0.10f * amag;   // ~0.1 s: slow movement is removed, fast tremor stays
    gx_avg = 0.90f * gx_avg + 0.10f * gx;
    gy_avg = 0.90f * gy_avg + 0.10f * gy;
    gz_avg = 0.90f * gz_avg + 0.10f * gz;

    float a_hp = amag - g_avg;
    float dgx = gx - gx_avg, dgy = gy - gy_avg, dgz = gz - gz_avg;

    xSemaphoreTake(tremor_mutex, portMAX_DELAY);
    trem_acc_sq  += a_hp * a_hp;
    trem_gyro_sq += dgx * dgx + dgy * dgy + dgz * dgz;
    trem_count++;
    xSemaphoreGive(tremor_mutex);
  }
}

// Tremor level for the last 50 ms window, 0-1 (called by sensorTask)
float takeTremorLevel() {
  double a_sq, g_sq; uint32_t n;
  xSemaphoreTake(tremor_mutex, portMAX_DELAY);
  a_sq = trem_acc_sq; g_sq = trem_gyro_sq; n = trem_count;
  trem_acc_sq = 0; trem_gyro_sq = 0; trem_count = 0;
  xSemaphoreGive(tremor_mutex);
  if (n == 0) return last_tremor;                   // no new MPU sample yet: hold last value
  float a_rms = sqrt(a_sq / n);
  float g_rms = sqrt(g_sq / n);
  float lvl = fmaxf(a_rms / TREMOR_ACC_FS_G, g_rms / TREMOR_GYRO_FS_DPS);
  last_tremor = constrain(lvl, 0.0f, 1.0f);
  return last_tremor;
}

// ============================================================
//  TASK: SENSOR (Core 1, priority 4) — 20 Hz
// ============================================================
void sensorTask(void* param) {
  for (;;) {
    if (xSemaphoreTake(data_ready_sem, portMAX_DELAY) == pdTRUE) {
      float rf = read_normalized(FSR_PIN);
      float rg = read_normalized(GSR_PIN);

      ema_fsr = ema_filter(ema_fsr, rf);
      ema_gsr = ema_filter(ema_gsr, rg);
      if (mpu_ok) {
        // light smoothing only: the RMS window already averages, and heavy
        // smoothing would blur short tremor bursts
        ema_trem = 0.6f * takeTremorLevel() + 0.4f * ema_trem;
      } else {
        ema_trem = ema_filter(ema_trem, read_normalized(TREMOR_PIN));
      }

      ProcessedSample_t p;
      p.fsr_norm     = ema_fsr;
      p.gsr_norm     = ema_gsr;
      p.tremor_norm  = ema_trem;
      p.stress_score = 0;

      xSemaphoreTake(latest_proc_mutex, portMAX_DELAY);
      latest_proc = p;
      xSemaphoreGive(latest_proc_mutex);
    }
  }
}

// ============================================================
//  PARSE COMMANDS FROM PYTHON (unchanged)
// ============================================================
void parseIncoming(String line) {
  line.trim();
  if (line.startsWith("ALERT:")) {
    int lvl = line.substring(6).toInt();
    lvl = constrain(lvl, 0, 3);
    xSemaphoreTake(stress_mutex, portMAX_DELAY);
    alert_level = lvl;
    xSemaphoreGive(stress_mutex);
    digitalWrite(LED_PIN, lvl >= 2 ? HIGH : LOW);
    return;
  }
  if (line.startsWith("DISP:")) {
    String p = line.substring(5);
    int d1 = p.indexOf('|');
    int d2 = p.indexOf('|', d1 + 1);
    int d3 = p.indexOf('|', d2 + 1);
    if (d1 > 0 && d2 > 0 && d3 > 0) {
      xSemaphoreTake(stress_mutex, portMAX_DELAY);
      state_label   = p.substring(0, d1);
      confidence    = p.substring(d1 + 1, d2).toFloat();
      fatigue_score = p.substring(d2 + 1, d3).toFloat();
      gci_val       = p.substring(d3 + 1).toFloat();
      xSemaphoreGive(stress_mutex);
    }
    return;
  }
}

// ============================================================
//  TASK: COMMS (Core 0, priority 3) (unchanged)
// ============================================================
void commsTask(void* param) {
  unsigned long last_send = 0, last_hb = 0;
  uint32_t count = 0;
  String inBuf = "";
  for (;;) {
    unsigned long now = millis();
    if (now - last_send >= (1000 / SAMPLE_RATE_HZ)) {
      last_send = now;
      xSemaphoreTake(latest_proc_mutex, portMAX_DELAY);
      ProcessedSample_t p = latest_proc;
      xSemaphoreGive(latest_proc_mutex);
      Serial.printf("%.4f %.4f %.4f\n", p.fsr_norm, p.gsr_norm, p.tremor_norm);
      count++;
    }
    if (now - last_hb >= 5000) {
      Serial.print("HB "); Serial.println(count);
      last_hb = now;
    }
    while (Serial.available()) {
      char c = (char)Serial.read();
      if (c == '\n') {
        if (inBuf.length() > 0) { parseIncoming(inBuf); inBuf = ""; }
      } else {
        inBuf += c;
        if (inBuf.length() > 80) inBuf = "";
      }
    }
    vTaskDelay(pdMS_TO_TICKS(5));
  }
}

// ============================================================
//  TASK: BUZZER (Core 1, priority 1) (unchanged behaviour)
// ============================================================
void buzzerTask(void* param) {
  bool siren_toggle = false;
  unsigned long last_beep = 0;
  for (;;) {
    unsigned long now = millis();
    xSemaphoreTake(stress_mutex, portMAX_DELAY);
    int al = alert_level;
    xSemaphoreGive(stress_mutex);
    switch (al) {
      case 0: noTone(BUZZER_PIN); break;
      case 1:
        if (now - last_beep > 3000) { tone(BUZZER_PIN, 800, 200); last_beep = now; }
        break;
      case 2:
        if (now - last_beep > 1500) {
          tone(BUZZER_PIN, 1200, 120);
          vTaskDelay(pdMS_TO_TICKS(200));
          tone(BUZZER_PIN, 1200, 120);
          last_beep = now;
        }
        break;
      case 3:
        if (now - last_beep > 350) {
          tone(BUZZER_PIN, siren_toggle ? 2200 : 700, 320);
          siren_toggle = !siren_toggle;
          last_beep = now;
        }
        break;
    }
    vTaskDelay(pdMS_TO_TICKS(20));
  }
}

// ============================================================
//  OLED
// ============================================================
const char* ALERT_BANNER[] = {"  SAFE  ", "CAUTION ", "WARNING ", "CRITICAL"};

void updateOLED() {
  if (!oled_ok) return;

  xSemaphoreTake(stress_mutex, portMAX_DELAY);
  int al = alert_level; String st = state_label;
  float conf = confidence, score = fatigue_score, gci = gci_val;
  xSemaphoreGive(stress_mutex);

  xSemaphoreTake(latest_proc_mutex, portMAX_DELAY);
  float fsr = latest_proc.fsr_norm, trem = latest_proc.tremor_norm;
  xSemaphoreGive(latest_proc_mutex);

  // drawing happens in RAM; only display.display() uses the I2C bus
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(2);
  if (al == 3) {
    display.fillRect(0, 0, 128, 17, SSD1306_WHITE);
    display.setTextColor(SSD1306_BLACK);
  }
  display.setCursor(0, 0);
  display.print(ALERT_BANNER[al]);
  display.setTextColor(SSD1306_WHITE);
  display.drawLine(0, 18, 128, 18, SSD1306_WHITE);
  display.setTextSize(1);

  if (al == 0 || al == 1) {
    display.setCursor(0, 21);
    display.print("State: "); display.print(st.substring(0, 9));
    display.setCursor(0, 31);
    display.print("Conf:  "); display.print((int)(conf * 100));
    display.print("%   Score:"); display.print((int)score);
    display.setCursor(0, 43);
    display.print("GCI:");
    int bar = constrain((int)(gci * 70.0f), 0, 70);
    display.drawRect(28, 43, 71, 7, SSD1306_WHITE);
    if (bar > 0) display.fillRect(28, 43, bar, 7, SSD1306_WHITE);
    display.setCursor(0, 55);
    display.print("Grip:"); display.print((int)(fsr * 100));
    display.print("% Trem:"); display.print((int)(trem * 100));
    display.print(mpu_ok ? "%" : "%p");    // "p" = tremor is the pot fallback
  } else if (al == 2) {
    display.setTextSize(2);
    display.setCursor(4, 22);  display.print("SLOW DOWN");
    display.setCursor(0, 42);  display.print("TAKE A BREAK");
  } else {
    display.setTextSize(2);
    display.setCursor(10, 22); display.print("!! STOP !!");
    display.setCursor(8, 42);  display.print("THE VEHICLE");
  }

  xSemaphoreTake(i2c_mutex, portMAX_DELAY);
  display.display();
  xSemaphoreGive(i2c_mutex);
}

void oledTask(void* param) {
  for (;;) {
    updateOLED();
    vTaskDelay(pdMS_TO_TICKS(250));
  }
}

void splashScreen() {
  if (!oled_ok) return;
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(2);
  display.setCursor(4, 2);   display.print("NeuroGrip");
  display.setTextSize(1);
  display.setCursor(10, 22); display.print("Driver Safety v3.0");
  display.setCursor(14, 34); display.print("Janavi & Hardik");
  display.setCursor(10, 46); display.print(mpu_ok ? "MPU6050: OK" : "MPU6050: not found");
  display.display();
  delay(2500);
}

// ============================================================
//  SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  // FreeRTOS objects first (the I2C mutex is needed below)
  latest_proc_mutex = xSemaphoreCreateMutex();
  stress_mutex      = xSemaphoreCreateMutex();
  i2c_mutex         = xSemaphoreCreateMutex();
  tremor_mutex      = xSemaphoreCreateMutex();
  data_ready_sem    = xSemaphoreCreateBinary();

  // I2C devices (no tasks running yet, so no mutex needed here)
  Wire.begin(21, 22);
  Wire.setClock(400000);
  mpu_ok  = mpuInit();
  oled_ok = display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
  Serial.println(mpu_ok ? "MPU_OK" : "MPU_MISSING_USING_POT");
  if (!oled_ok) Serial.println("OLED_MISSING");
  else splashScreen();

  // 20 Hz sampling timer (API differs between ESP32 core 2.x and 3.x)
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  sample_timer = timerBegin(1000000);                        // 1 MHz tick
  timerAttachInterrupt(sample_timer, &onSampleTimer);
  timerAlarm(sample_timer, 1000000 / SAMPLE_RATE_HZ, true, 0);
#else
  sample_timer = timerBegin(0, 80, true);                    // 1 MHz tick
  timerAttachInterrupt(sample_timer, &onSampleTimer, true);
  timerAlarmWrite(sample_timer, 1000000 / SAMPLE_RATE_HZ, true);
  timerAlarmEnable(sample_timer);
#endif

  // startup chime
  tone(BUZZER_PIN, 880, 100);  delay(150);
  tone(BUZZER_PIN, 1175, 100); delay(150);
  tone(BUZZER_PIN, 1568, 150); delay(300);

  // Core 1: sensors, MPU, buzzer, OLED.  Core 0: serial.
  xTaskCreatePinnedToCore(sensorTask, "Sensor", 3072, NULL, 4, NULL, 1);
  if (mpu_ok) xTaskCreatePinnedToCore(mpuTask, "MPU", 3072, NULL, 3, NULL, 1);
  xTaskCreatePinnedToCore(buzzerTask, "Buzzer", 2048, NULL, 1, NULL, 1);
  xTaskCreatePinnedToCore(oledTask,   "OLED",   3072, NULL, 1, NULL, 1);
  xTaskCreatePinnedToCore(commsTask,  "Comms",  4096, NULL, 3, NULL, 0);

  Serial.println("NEUROGRIP_READY");
}

void loop() {
  vTaskDelay(portMAX_DELAY);
}