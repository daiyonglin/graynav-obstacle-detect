# GrayNav unified board build evidence (2026-08-13)

## Status boundary

The unified Indoor8 + Scene21 model has completed official A1 INT8 conversion,
single-model board integration, host postprocess tests, A1 cross compilation,
and initramfs inspection. The resulting image is a **flash candidate**. It has
not yet passed physical-board inference, Aurora, serial, voice, or endurance
acceptance, so it must not be described as deployed or validated.

## Model contract

```text
input  images        1 x 1 x 384 x 384

out0   cls_p3        1 x  8 x 48 x 48
out1   reg_p3        1 x 64 x 48 x 48
out2   cls_p4        1 x  8 x 24 x 24
out3   reg_p4        1 x 64 x 24 x 24
out4   cls_p5        1 x  8 x 12 x 12
out5   reg_p5        1 x 64 x 12 x 12
out6   scene_logits  1 x 21 x 48 x 48
```

The runtime validates this exact order, channel count, grid size, element
count, and dtype-compatible tensor view before decoding. A mismatch invalidates
the whole inference and displays `AI_FAIL`; stale scene/depth evidence is not
used for navigation.

## Official conversion evidence

```text
m1model bytes   4,150,950
m1model SHA256  33EEC832710706B1153F468F219C08389A52BA3D21CBDFFCDE32CA5E25D66DA8
```

| order | output | overall cosine | minimum per-sample cosine |
|---:|---|---:|---:|
| 0 | cls_p3 | 0.994585 | 0.991113 |
| 1 | reg_p3 | 0.963706 | 0.949598 |
| 2 | cls_p4 | 0.991130 | 0.986160 |
| 3 | reg_p4 | 0.941258 | 0.916017 |
| 4 | cls_p5 | 0.990634 | 0.986307 |
| 5 | reg_p5 | 0.968374 | 0.935184 |
| 6 | scene_logits | 0.969735 | 0.950890 |

`reg_p4` is the weakest quantized branch. Board qualification must explicitly
watch the stability of medium-size object boxes, while scene output is judged
by PATH/BLOCKED/STAIR behavior rather than raw logits.

## Build and rootfs audit

Build command:

```powershell
docker exec A1_Builder sh -lc `
  'cd /home/smartsens_flying_chip_a1_sdk/A1_SDK_SC132GS/smartsens_sdk && ./scripts/a1_sc132gs_build.sh'
```

Verified Buildroot/CMake contract:

```text
A1_YOLO_NUM_CLASSES=8
A1_YOLO_INPUT_CHANNELS=1
A1_ENABLE_VOICE=ON
A1_REQUIRE_MODEL=ON
A1_MODEL_FILENAME=graynav_unified_indoor8_scene21.m1model
```

Verified rootfs content:

```text
.m1model count       1
fixed .ssbmp count   17
UART kernel module   present
run.sh               present
```

Final candidate:

```text
file     zImage.smartsens-m1-evb
bytes    8,106,192
SHA256   DD1E9C49DD6BA51A2F159013BC532FA081AEEE851DA545AC1C001BB678C904BA
limit    < 15 MiB (passed)
```

External immutable candidate archive:

```text
E:\jichuang\graynav_firmware_archive\unified_indoor8_scene21_2026-08-13_DD1E9C49
```

Protected rollback image remains unchanged:

```text
bytes    8,214,488
SHA256   A7976710ECB456CB312D18F0195DCAE496ED652EFC582AB698EBC3EB7B055530
```

## Demo behavior implemented

- One `model_id`; every NPU inference produces all seven outputs.
- ROI schedule is `LOWER -> LOWER -> UPPER`. Only LOWER advances road and
  stair temporal state; UPPER refreshes object detection without contaminating
  floor-scene history.
- OSD budget is fixed: one action texture, one primary label, at most eight
  scene primitives, and at most three stable boxes. No dense text, point bars,
  color masks, or legacy risk tiles are installed in the rootfs.
- Primary label priority is `STAIR > BLOCKED > object > PATH > UNKNOWN`.
- Serial state changes print immediately; stable state prints at most once per
  second using the compact `[NAV]` record.
- Voice defaults enabled and remains asynchronous. `CLEAR` is announced once
  on recovery and has no periodic repeat; ordinary hazards use a five-second
  cooldown, STOP can repeat after two seconds, and UART failure does not stop
  video or inference.

## Required physical-board qualification

1. Confirm startup reports one model ID, one input, seven outputs, and the exact
   `out0..out6` contract above. Any `[UNIFIED][ERROR]` is a release blocker.
2. Confirm Aurora shows no black-dot glyph noise and no OSD add/flush failures.
3. Exercise full person, face/upper body, back/side body, legs, chair, table,
   bag, suitcase, couch/bench, flat floor, wall, real stairs, stair-like edges,
   glare, dark scene, and lens cover.
4. Confirm the primary static label, shapes, `[NAV]` line, and SYN6288 phrase
   agree on the same cause. Keep tests supervised; do not walk blindfolded.
5. Record at least 30 minutes of serial output and verify no crash, sustained
   memory growth, repeated CLEAR speech, or OSD add/flush failure.

Only after this qualification may the image status change from `built` to
`board-tested`.
