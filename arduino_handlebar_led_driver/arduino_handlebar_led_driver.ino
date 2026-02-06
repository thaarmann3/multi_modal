/*
  Read 16 RGB pixels from Serial in this format (one line):
    "0,0,255;0,255,0;255,0,0;... (16 total triples)"

  - Values are clamped to 0..255
  - NeoPixel strip is on pin 3, 16 pixels
  - Send a newline '\n' at the end of the line
*/

#include <Adafruit_NeoPixel.h>

static const uint8_t  PIN_NEOPIXEL = 3;
static const uint16_t NUM_PIXELS   = 16;

Adafruit_NeoPixel strip(NUM_PIXELS, PIN_NEOPIXEL, NEO_GRB + NEO_KHZ800);

// Serial line buffer
static const uint16_t LINE_BUF_SIZE = 256;
char lineBuf[LINE_BUF_SIZE];
uint16_t lineLen = 0;

static inline uint8_t clampU8(int v) {
  if (v < 0) return 0;
  if (v > 255) return 255;
  return (uint8_t)v;
}

static void skipWs(const char* &p) {
  while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
}

static bool parseIntVal(const char* &p, int &out) {
  skipWs(p);

  bool neg = false;
  if (*p == '-') { neg = true; p++; }

  if (*p < '0' || *p > '9') return false;

  long v = 0;
  while (*p >= '0' && *p <= '9') {
    v = v * 10 + (*p - '0');
    p++;
  }
  out = neg ? (int)-v : (int)v;
  return true;
}

static bool expectChar(const char* &p, char expected) {
  skipWs(p);
  if (*p != expected) return false;
  p++;
  return true;
}

bool parse16RGB(const char* s, uint8_t rgbOut[NUM_PIXELS][3]) {
  const char* p = s;

  for (uint16_t i = 0; i < NUM_PIXELS; i++) {
    int r, g, b;

    if (!parseIntVal(p, r)) return false;
    if (!expectChar(p, ',')) return false;

    if (!parseIntVal(p, g)) return false;
    if (!expectChar(p, ',')) return false;

    if (!parseIntVal(p, b)) return false;

    rgbOut[i][0] = clampU8(r);
    rgbOut[i][1] = clampU8(g);
    rgbOut[i][2] = clampU8(b);

    // Pixel separator: ';' between pixels (optional trailing ';' at end)
    skipWs(p);
    if (i < NUM_PIXELS - 1) {
      if (!expectChar(p, ';')) return false;
    } else {
      // last pixel: allow end, or a trailing ';'
      if (*p == ';') p++;
      skipWs(p);
      if (*p != '\0') return false; // unexpected extra characters
    }
  }

  return true;
}

void applyRGB(const uint8_t rgb[NUM_PIXELS][3]) {
  for (uint16_t i = 0; i < NUM_PIXELS; i++) {
    strip.setPixelColor(i, strip.Color(rgb[i][0], rgb[i][1], rgb[i][2]));
  }
  strip.show();
}

void setup() {
  Serial.begin(250000);
  strip.begin();
  strip.show();          // clear
  strip.setBrightness(255); // keep global brightness max; use per-pixel RGB directly

  Serial.println(F("Ready. Send 16 RGB triples: r,g,b;r,g,b;... (newline terminated)"));
  Serial.println(F("Example: 0,0,255;0,255,0;255,0,0; ..."));
}

void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\n') {
      lineBuf[lineLen] = '\0';

      uint8_t rgb[NUM_PIXELS][3];
      if (parse16RGB(lineBuf, rgb)) {
        applyRGB(rgb);
      } else {
        Serial.println(F("ERR: expected 16 triples like r,g,b;r,g,b;... (values 0-255)"));
      }

      lineLen = 0;
      return;
    }

    if (c == '\r') continue; // ignore CR

    if (lineLen < LINE_BUF_SIZE - 1) {
      lineBuf[lineLen++] = c;
    } else {
      lineLen = 0;
      Serial.println(F("ERR: line too long"));
    }
  }
}
