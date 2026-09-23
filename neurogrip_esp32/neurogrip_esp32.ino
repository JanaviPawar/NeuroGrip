#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/semphr.h>

// ============================================================
//  PIN DEFINITIONS
// ============================================================
#define FSR_PIN     34
#define GSR_PIN     35
#define TREMOR_PIN  32
#define BUZZER_PIN  25
#define LED_PIN      2

// ============================================================
//  OLED
// ============================================================
#define OLED_W      128
#define OLED_H       64
#define OLED_RESET   -1
#define OLED_ADDR  0x3C
Adafruit_SSD1306 display(OLED_W, OLED_H, &Wire, OLED_RESET);
bool oled_ok = false;

// ============================================================
//  SAMPLING
// ============================================================
#define SAMPLE_RATE_HZ  20
#define EMA_ALPHA       0.35f

// ============================================================
//  SHARED STATE  (all protected by stress_mutex)
// ============================================================
volatile int   alert_level  = 0;     // 0=SAFE 1=CAUTION 2=WARNING 3=CRITICAL
String         state_label  = "Relax";
float          confidence   = 0.0f;
float          fatigue_score = 0.0f;
float          gci_val      = 1.0f;

// ============================================================
//  FREERTOS OBJECTS
// ============================================================
typedef struct {
  float fsr_norm, gsr_norm, tremor_norm;
  float stress_score;
} ProcessedSample_t;

ProcessedSample_t latest_proc = {0};
SemaphoreHandle_t latest_proc_mutex;
SemaphoreHandle_t stress_mutex;
SemaphoreHandle_t data_ready_sem;
hw_timer_t*       sample_timer = NULL;

// ============================================================
//  HARDWARE TIMER ISR
// ============================================================
void IRAM_ATTR onSampleTimer() {
  BaseType_t xHigher = pdFALSE;
  xSemaphoreGiveFromISR(data_ready_sem, &xHigher);
  if (xHigher) portYIELD_FROM_ISR();
}

// ============================================================
//  ADC READ  (oversampled, normalized 0–1)
// ============================================================
inline float read_normalized(int pin) {
  uint32_t sum = 0;
  for (int i = 0; i < 4; i++) {
    sum += analogRead(pin);
    delayMicroseconds(30);
  }
  return (sum / 4) / 4095.0f;
}

// ============================================================
//  EMA FILTER
// ============================================================
float ema_fsr = 0, ema_gsr = 0, ema_trem = 0;
inline float ema_filter(float prev, float nv) {
  return EMA_ALPHA * nv + (1.0f - EMA_ALPHA) * prev;
}

// ============================================================
//  TASK 1: SENSOR TASK  (Core 1, Priority 4)
// ============================================================
void sensorTask(void* param) {
  for (;;) {
    if (xSemaphoreTake(data_ready_sem, portMAX_DELAY) == pdTRUE) {
      float rf = read_normalized(FSR_PIN);
      float rg = read_normalized(GSR_PIN);
      float rt = read_normalized(TREMOR_PIN);

      ema_fsr  = ema_filter(ema_fsr,  rf);
      ema_gsr  = ema_filter(ema_gsr,  rg);
      ema_trem = ema_filter(ema_trem, rt);

      ProcessedSample_t p;
      p.fsr_norm    = ema_fsr;
      p.gsr_norm    = ema_gsr;
      p.tremor_norm = ema_trem;
      p.stress_score = 0;

      xSemaphoreTake(latest_proc_mutex, portMAX_DELAY);
      latest_proc = p;
      xSemaphoreGive(latest_proc_mutex);
    }
  }
}

// ============================================================
//  PARSE INCOMING FROM PYTHON
//  Called inside commsTask — runs on Core 0
//
//  Python sends two commands per cycle:
//    "ALERT:2\n"                          → sets alert_level
//    "DISP:Stressed|0.87|45.2|0.73\n"    → updates OLED fields
// ============================================================
void parseIncoming(String line) {
  line.trim();

  // ── ALERT command ──────────────────────────────────────
  if (line.startsWith("ALERT:")) {
    int lvl = line.substring(6).toInt();
    lvl = constrain(lvl, 0, 3);
    xSemaphoreTake(stress_mutex, portMAX_DELAY);
    alert_level = lvl;
    xSemaphoreGive(stress_mutex);
    digitalWrite(LED_PIN, lvl >= 2 ? HIGH : LOW);
    return;
  }

  // ── DISP command ───────────────────────────────────────
  // Format: "DISP:State|Conf|Score|GCI"
  if (line.startsWith("DISP:")) {
    String p  = line.substring(5);
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
//  TASK 2: COMMS TASK  (Core 0, Priority 3)
//  Sends sensor data to Python + reads ALERT/DISP commands back
// ============================================================
void commsTask(void* param) {
  unsigned long last_send = 0;
  uint32_t      count     = 0;
  unsigned long last_hb   = 0;
  String        inBuf     = "";

  for (;;) {
    unsigned long now = millis();

    // ── Send sensor data at SAMPLE_RATE_HZ ──────────────
    if (now - last_send >= (1000 / SAMPLE_RATE_HZ)) {
      last_send = now;

      xSemaphoreTake(latest_proc_mutex, portMAX_DELAY);
      ProcessedSample_t p = latest_proc;
      xSemaphoreGive(latest_proc_mutex);

      Serial.printf("%.4f %.4f %.4f\n", p.fsr_norm, p.gsr_norm, p.tremor_norm);
      count++;
    }

    // ── Heartbeat every 5s ───────────────────────────────
    if (now - last_hb >= 5000) {
      Serial.print("HB "); Serial.println(count);
      last_hb = now;
    }

    // ── READ incoming commands from Python ───────────────
    // CRITICAL FIX: was `while(Serial.available()) Serial.read();`
    // That discarded every ALERT: command Python sent — buzzer never fired.
    // Now we buffer until newline and parse properly.
    while (Serial.available()) {
      char c = (char)Serial.read();
      if (c == '\n') {
        if (inBuf.length() > 0) {
          parseIncoming(inBuf);
          inBuf = "";
        }
      } else {
        inBuf += c;
        if (inBuf.length() > 80) inBuf = "";  // overflow guard
      }
    }

    vTaskDelay(pdMS_TO_TICKS(5));
  }
}

// ============================================================
//  TASK 3: BUZZER TASK  (Core 1, Priority 1)
//
//  alert_level 0 → silent
//  alert_level 1 → single beep every 3s   (CAUTION)
//  alert_level 2 → double beep every 1.5s (WARNING)
//  alert_level 3 → alternating siren      (CRITICAL)
// ============================================================
void buzzerTask(void* param) {
  bool          siren_toggle = false;
  unsigned long last_beep    = 0;

  for (;;) {
    unsigned long now = millis();

    xSemaphoreTake(stress_mutex, portMAX_DELAY);
    int al = alert_level;
    xSemaphoreGive(stress_mutex);

    switch (al) {
      case 0:
        noTone(BUZZER_PIN);
        break;

      case 1:
        // CAUTION — gentle single beep every 3 seconds
        if (now - last_beep > 3000) {
          tone(BUZZER_PIN, 800, 200);
          last_beep = now;
        }
        break;

      case 2:
        // WARNING — urgent double beep every 1.5 seconds
        if (now - last_beep > 1500) {
          tone(BUZZER_PIN, 1200, 120);
          vTaskDelay(pdMS_TO_TICKS(200));
          tone(BUZZER_PIN, 1200, 120);
          last_beep = now;
        }
        break;

      case 3:
        // CRITICAL — alternating siren
        if (now - last_beep > 350) {
          tone(BUZZER_PIN, siren_toggle ? 2200 : 700, 320);
          siren_toggle = !siren_toggle;
          last_beep    = now;
        }
        break;
    }

    vTaskDelay(pdMS_TO_TICKS(20));
  }
}

// ============================================================
//  OLED UPDATE
//
//  Layout (128×64):
//  Line 0 (y=0):  Alert banner — "  SAFE  " / "CAUTION " /
//                               "WARNING " / "CRITICAL" (inverted on CRITICAL)
//  Line 1 (y=20): State from Python  e.g. "State: Stressed"
//  Line 2 (y=30): Confidence         e.g. "Conf:  87%"
//  Line 3 (y=40): BIG MESSAGE:
//                   al=2 → "SLOW DOWN"  "TAKE A BREAK"
//                   al=3 → "!! STOP !!"  "THE VEHICLE"
//  Line 4 (y=55): GCI bar / Calibrating
// ============================================================
const char* ALERT_BANNER[] = {"  SAFE  ", "CAUTION ", "WARNING ", "CRITICAL"};

void updateOLED() {
  if (!oled_ok) return;

  // Snapshot shared state (mutex)
  xSemaphoreTake(stress_mutex, portMAX_DELAY);
  int   al    = alert_level;
  String st   = state_label;
  float conf  = confidence;
  float score = fatigue_score;
  float gci   = gci_val;
  xSemaphoreGive(stress_mutex);

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  // ── Alert banner (top, size 2) ────────────────────────
  display.setTextSize(2);
  if (al == 3) {
    // Inverted fill for CRITICAL
    display.fillRect(0, 0, 128, 17, SSD1306_WHITE);
    display.setTextColor(SSD1306_BLACK);
  }
  display.setCursor(0, 0);
  display.print(ALERT_BANNER[al]);
  display.setTextColor(SSD1306_WHITE);
  display.drawLine(0, 18, 128, 18, SSD1306_WHITE);

  // ── State + confidence (size 1) ──────────────────────
  display.setTextSize(1);

  if (al == 0 || al == 1) {
    // Normal info display
    display.setCursor(0, 21);
    display.print("State: ");
    display.print(st.substring(0, 9));

    display.setCursor(0, 31);
    display.print("Conf:  ");
    display.print((int)(conf * 100));
    display.print("%   Score:");
    display.print((int)score);

    // GCI bar
    display.setCursor(0, 43);
    display.print("GCI:");
    int bar = (int)(gci * 70.0f);
    bar = constrain(bar, 0, 70);
    display.drawRect(28, 43, 71, 7, SSD1306_WHITE);
    if (bar > 0) display.fillRect(28, 43, bar, 7, SSD1306_WHITE);

    display.setCursor(0, 55);
    display.print("Bio:");
    // Show FSR and GSR values from latest_proc
    xSemaphoreTake(latest_proc_mutex, portMAX_DELAY);
    float fsr = latest_proc.fsr_norm;
    float gsr = latest_proc.gsr_norm;
    xSemaphoreGive(latest_proc_mutex);
    display.print((int)(fsr * 100));
    display.print("%  GSR:");
    display.print((int)(gsr * 100));
    display.print("%");

  } else if (al == 2) {
    // ── WARNING: "SLOW DOWN  TAKE A BREAK" ──────────────
    // Big text in centre
    display.setTextSize(2);
    display.setCursor(4, 22);
    display.print("SLOW DOWN");
    display.setCursor(0, 42);
    display.print("TAKE A BREAK");

  } else if (al == 3) {
    // ── CRITICAL: "!! STOP !!  THE VEHICLE" ─────────────
    display.setTextSize(2);
    display.setCursor(10, 22);
    display.print("!! STOP !!");
    display.setCursor(8, 42);
    display.print("THE VEHICLE");
  }

  display.display();
}

// ============================================================
//  TASK 4: OLED TASK  (Core 1, Priority 1)
//  Refreshes display every 250ms — independent of comms
// ============================================================
void oledTask(void* param) {
  for (;;) {
    updateOLED();
    vTaskDelay(pdMS_TO_TICKS(250));
  }
}

// ============================================================
//  SPLASH SCREEN
// ============================================================
void splashScreen() {
  if (!oled_ok) return;
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(2);
  display.setCursor(4, 2);
  display.print("NeuroGrip");
  display.setTextSize(1);
  display.setCursor(10, 22);
  display.print("Driver Safety v2.5");
  display.setCursor(14, 34);
  display.print("Janavi & Hardik");
  display.setCursor(18, 46);
  display.print("Initializing...");
  display.display();
  delay(2500);
}

// ============================================================
//  SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN,    OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  // ── OLED init ────────────────────────────────────────────
  Wire.begin(21, 22);
  oled_ok = display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
  if (!oled_ok) {
    Serial.println("OLED not found — continuing without display");
  } else {
    splashScreen();
  }

  // ── FreeRTOS objects ─────────────────────────────────────
  latest_proc_mutex = xSemaphoreCreateMutex();
  stress_mutex      = xSemaphoreCreateMutex();
  data_ready_sem    = xSemaphoreCreateBinary();

  // ── Hardware timer (sampling ISR) ────────────────────────
  sample_timer = timerBegin(0, 80, true);           // 1 MHz tick
  timerAttachInterrupt(sample_timer, &onSampleTimer, true);
  timerAlarmWrite(sample_timer, 1000000 / SAMPLE_RATE_HZ, true);
  timerAlarmEnable(sample_timer);

  // ── Startup chime (confirms buzzer is wired correctly) ───
  tone(BUZZER_PIN, 880,  100); delay(150);
  tone(BUZZER_PIN, 1175, 100); delay(150);
  tone(BUZZER_PIN, 1568, 150); delay(300);

  // ── FreeRTOS Tasks ───────────────────────────────────────
  // Core 1: sensors, buzzer, OLED (real-time side)
  xTaskCreatePinnedToCore(sensorTask, "Sensor",  2048, NULL, 4, NULL, 1);
  xTaskCreatePinnedToCore(buzzerTask, "Buzzer",  1024, NULL, 1, NULL, 1);
  xTaskCreatePinnedToCore(oledTask,   "OLED",    2048, NULL, 1, NULL, 1);

  // Core 0: serial comms with Python
  xTaskCreatePinnedToCore(commsTask,  "Comms",   4096, NULL, 3, NULL, 0);

  Serial.println("NEUROGRIP_READY");
}

// ============================================================
//  LOOP — FreeRTOS handles everything
// ============================================================
void loop() {
  vTaskDelay(portMAX_DELAY);
}
