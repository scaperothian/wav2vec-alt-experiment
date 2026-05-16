BASE_MODEL = "facebook/wav2vec2-large-960h-lv60-self"
TARGET_SR = 16_000
EXPECTED_HIDDEN = 1024

WINDOW_SEC = 1.0
HOP_SEC = 0.5

# wav2vec2-large CNN feature extractor: stride = 5*2^6 = 320 at 16 kHz
WAV2VEC_FRAME_RATE = 50

# wav2vec2-large has 24 transformer layers; probe every 6th
LAYERS_TO_PROBE = [6, 12, 18, 24]
