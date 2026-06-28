# UART Voice Module Integration

The current system supports three output modes:

- `A1_OUTPUT_MODE=osd`: Aurora OSD only. This is the default.
- `A1_OUTPUT_MODE=voice`: UART voice output only, no OSD refresh.
- `A1_OUTPUT_MODE=both`: Aurora OSD and UART voice output at the same time.

## Recommended P4 Wiring

Use A1-side UART pins, not the CH347 USB-debug bridge pins.

| Signal | P4 pin | Connect to voice module |
| --- | ---: | --- |
| `A1_D1_UART1TX` | 4 | module `RX` |
| `A1_D3_UART1RX` | 6 | module `TX`, optional for ACK/debug |
| `GND` | 33/34/47/48 | module `GND` |

Use a 3.3 V TTL UART voice module. Do not connect a 5 V TTL output directly to
A1 UART RX. If the voice module needs 5 V power, power it separately and keep
the UART level at 3.3 V or add level shifting.

Avoid `CH347_UART_TX/RX` on P4 pins 3/5 for the final voice module. Those pins
belong to the USB debug bridge path and may conflict with PC serial debugging.

## Board Runtime

Default OSD-only run:

```sh
./scripts/run.sh
```

OSD + voice:

```sh
A1_OUTPUT_MODE=both A1_VOICE_UART=/dev/ttyS1 A1_VOICE_BAUD=115200 ./scripts/run_voice_both.sh
```

Voice-only demonstration:

```sh
A1_OUTPUT_MODE=voice A1_VOICE_UART=/dev/ttyS1 ./scripts/run_voice_both.sh
```

If `/dev/ttyS1` is not UART1 on the board, list candidates:

```sh
ls -l /dev/ttyS* /dev/ttyAMA* 2>/dev/null
```

Then set `A1_VOICE_UART` to the working device.

## UART Protocol

The board sends ASCII lines terminated by `\r\n`.

Startup:

```text
HELLO,A1_OBSTACLE_V1
```

Navigation command:

```text
NAV,F=120,A=STOP,D=C,C=PERSON,R=URGENT,Z=NEAR,T=3
NAV,F=230,A=TURN_LEFT,D=C,C=CHAIR_SEAT,R=NEAR,Z=NEAR,T=7
NAV,F=360,A=SLOW,D=R,C=TABLE_DESK,R=WARNING,Z=WARN,T=9
NAV,F=480,A=CLEAR,D=C,C=NONE,R=UNK,Z=UNK,T=-1
```

Fields:

- `F`: frame id
- `A`: action, one of `STOP`, `TURN_LEFT`, `TURN_RIGHT`, `SLOW`, `CLEAR`
- `D`: direction, `L`, `C`, `R`, `LC`, `CR`, or `WIDE`
- `C`: semantic class
- `R`: risk level
- `Z`: distance bucket, `NEAR`, `WARN`, `FAR`, or `UNK`
- `T`: tracker id

Shutdown:

```text
BYE,A1_OBSTACLE_V1
```

## Arbitration

The voice notifier intentionally does not speak every detection frame:

- `STOP` can be sent immediately and can repeat after 1 second if still urgent.
- `TURN_LEFT`, `TURN_RIGHT`, and `SLOW` require a stable action for several
  update cycles.
- `CLEAR` requires longer stability.
- repeated identical prompts use a cooldown, default 4 seconds.

Environment variables:

```text
A1_VOICE_INTERVAL_FRAMES=5
A1_VOICE_STABLE_FRAMES=3
A1_VOICE_CLEAR_STABLE_FRAMES=18
A1_VOICE_COOLDOWN_MS=4000
```

The external module can either parse these ASCII commands directly or use a
small microcontroller to map commands to a UART TTS/MP3 voice module.

