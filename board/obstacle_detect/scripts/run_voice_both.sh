#!/bin/sh

chmod +x ./ssne_ai_demo

# Recommended hardware route:
#   A1 UART1 TX: P4 pin 4  (A1_D1_UART1TX) -> voice module RX
#   A1 UART1 RX: P4 pin 6  (A1_D3_UART1RX) -> voice module TX, optional
#   GND:         P4 pin 33/34/47/48        -> voice module GND
#
# Confirm the Linux device node on the board with:
#   ls -l /dev/ttyS* /dev/ttyAMA* 2>/dev/null
# If UART1 is not /dev/ttyS1 on the final image, override A1_VOICE_UART.

export A1_OUTPUT_MODE="${A1_OUTPUT_MODE:-both}"
export A1_VOICE_UART="${A1_VOICE_UART:-/dev/ttyS1}"
export A1_VOICE_BAUD="${A1_VOICE_BAUD:-115200}"
export A1_VOICE_INTERVAL_FRAMES="${A1_VOICE_INTERVAL_FRAMES:-5}"
export A1_VOICE_STABLE_FRAMES="${A1_VOICE_STABLE_FRAMES:-3}"
export A1_VOICE_CLEAR_STABLE_FRAMES="${A1_VOICE_CLEAR_STABLE_FRAMES:-18}"
export A1_VOICE_COOLDOWN_MS="${A1_VOICE_COOLDOWN_MS:-4000}"

./ssne_ai_demo

