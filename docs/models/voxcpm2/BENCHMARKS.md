# VoxCPM2 benchmarks

As of November 12, 2026, on Strix Halo at the `performance` profile (package
power 85 W median, 114 W peak, GPU edge at most 81 °C, logged every two
seconds in [artifacts/thermal.log](artifacts/thermal.log)):

- Gufo `f01a4570`, release build, with this extension (the revision of this
  document), loading the Hugging Face checkpoint (bfloat16) and audio.cpp's
  `voxcpm2-q8_0.gguf` (Q8_0 kept 8-bit on the GPU);
- audio.cpp server v0.9.4 (`f7ac446f`) on Vulkan with its own
  `voxcpm2-q8_0.gguf` and `voxcpm2-bf16.gguf`.

One server at a time, each started with a cold file cache.
`tools/bench/tts_bench.py` sends per variant one warm-up and three timed WAV
requests (the median counts) and one SSE request for the first audio, all
with seed 42:

- `short`: "The little brown dog ran across the garden."
- `auto`: the 37-word paragraph of Gufo's TTS benchmarks, automatic voice;
- `design`: the same paragraph after "(A young woman with a calm, low voice
  and a British accent)", VoxCPM2's voice-design syntax on both servers;
- `clone`: the paragraph in the voice of Gufo's test recording
  `clear-english-voice.wav` with its transcript, a preset on Gufo and a voice
  file on audio.cpp.

The real-time factor (RTF) is the WAV request's time divided by the audio's
length; lower is faster. The servers' audio differs in length (their noise
differs), so compare RTFs.

| Variant | Gufo, Q8_0 | audio.cpp Q8_0 | Gain | Gufo, bf16 | audio.cpp bf16 | Gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 0.192 | 0.360 | 1.88× | 0.310 | 0.754 | 2.43× |
| auto | 0.194 | 0.370 | 1.91× | 0.305 | 0.753 | 2.47× |
| design | 0.195 | 0.371 | 1.90× | 0.305 | 0.755 | 2.48× |
| clone | 0.205 | 0.370 | 1.80× | 0.314 | 0.759 | 2.42× |

With the same file, Gufo generates 1.80–1.91 times as fast as audio.cpp;
with bfloat16 weights, 2.42–2.48 times. Gufo's Q8_0 products keep the
activations in bfloat16, where audio.cpp quantizes them to 8 bits as well
([quality](QUALITY.md)).

First audio of the SSE stream:

| Variant | Gufo, Q8_0 | audio.cpp Q8_0 | Gufo, bf16 | audio.cpp bf16 |
| --- | ---: | ---: | ---: | ---: |
| short | 0.043 s | 0.086 s | 0.080 s | 0.185 s |
| auto | 0.052 s | 0.109 s | 0.108 s | 0.199 s |
| design | 0.054 s | 0.116 s | 0.109 s | 0.203 s |
| clone | 0.173 s | 0.592 s | 0.183 s | 0.951 s |

Readiness (from launch to the readiness endpoint, including Gufo's warm-up)
and memory after the requests:

| Server | Ready | Resident | GPU |
| --- | ---: | ---: | ---: |
| Gufo, Q8_0 | 7.10 s | 340 MiB | 4,064 MiB |
| audio.cpp Q8_0 | 4.79 s | 244 MiB | 6,567 MiB |
| Gufo, bf16 | 8.80 s | 301 MiB | 5,933 MiB |
| audio.cpp bf16 | 16.97 s | 221 MiB | 9,756 MiB |

- Repeated requests return identical audio on both servers. Gufo's PCM
  stream equals its WAV; audio.cpp's streaming model differs from its
  offline one, so its PCM doesn't.
- `gufo serve tts` caps the HIP runtime at one hardware queue
  (`GPU_MAX_HW_QUEUES=1`) unless the variable is set. AudioVAE's stream then
  shares the queue with the generator instead of decoding alongside it. With
  `GPU_MAX_HW_QUEUES=2` set before starting the server, the RTFs are
  0.185–0.197 with Q8_0 and 0.294–0.304 with bfloat16
  ([artifacts/gufo-q8_0-queues2.json](artifacts/gufo-q8_0-queues2.json),
  [artifacts/gufo-queues2.json](artifacts/gufo-queues2.json)), close to
  the 0.182 and 0.293 of `voxcpm_synthesize` for the same models and build.
- The products run on the matrix cores where they read 8-bit weights or the
  DiT's 22 rows; a DiT layer with Q8_0 reads its weights at 72 % of a plain
  read's rate ([experiments](EXPERIMENTS.md)).

The JSON results are in [artifacts](artifacts/).

## Reproduce

Run from a Gufo checkout with a release build that includes this extension:

```sh
VOXCPM_MODEL=/models/VoxCPM2 GUFO=build/release/gufo \
  python3 /path/to/voxcpm/tools/bench/tts_bench.py gufo gufo.json
AUDIOCPP_SERVER=/path/to/audiocpp_server \
AUDIOCPP_MODEL=/models/VoxCPM2-GGUF/voxcpm2-q8_0.gguf \
  python3 /path/to/voxcpm/tools/bench/tts_bench.py audiocpp audiocpp.json
```

`VOXCPM_MODEL` may also name an audio.cpp GGUF. `REFERENCE` names the clone's
WAV without its extension (default: the checkout's
`tests/functional/fixtures/audio/clear-english-voice`).
