# VoxCPM2 quality

As of November 12, 2026: Gufo `f01a4570`, ROCm 10.1, Strix Halo, the
`openbmb/VoxCPM2` checkpoint and audio.cpp's GGUFs, the extension's
`gpu-test` build.

## Against the reference implementation

`tools/reference/generator_golden.py` ran OpenBMB's VoxCPM2Model (revision
`300e426`) on the CPU in float32, the bfloat16 weights widened, and recorded
its tensors with seeded diffusion noise. The GPU tests feed the same prompts
and noise. Errors are relative L2 errors against the reference.

| Tensor | Zero-shot prompt | Reference-mode prompt |
| --- | ---: | ---: |
| Text LM, prefill | 2.9e-3 | 3.1e-3 |
| Residual LM, prefill | 2.2e-3 | 5.2e-3 |
| DiT condition, first patch | 2.0e-3 | 2.4e-3 |
| Stop logits, first patch | 1.8e-3 | 5.6e-4 |
| Latent, first patch (10 Euler steps, CFG 2.0) | 2.1e-3 | 4.7e-3 |
| Latent, second and third patch | 1.2e-2, 8.6e-3 | |
| One DiT evaluation (CFG pair, t = 0.63) | 7.6e-3 | |

The products take bfloat16 operands like the reference's GPU inference, with
float32 sums (`v_dot2_f32_bf16` for up to 32 rows, the matrix cores for the
DiT's 22 rows), norms and residual stream; that rounding explains the
deviation from the float32 run. Later patches inherit the earlier ones'
deviations through the autoregressive state: by the third patch the text
LM's state differs by 0.7–1.4 % (it reached 5.9 % with earlier kernels), so
the test allows 10 % after the first patch.

AudioVAE V2 runs in float32 like the reference: decoding the three zero-shot
patches, patch by patch with the carried convolution state or at once,
differs by 1.1e-6; encoding a 16 kHz test signal by 1.1e-6.

From audio.cpp's GGUFs (the `gguf` modes of the GPU tests; their limits
are five times wider):

| File | Generator | AudioVAE |
| --- | --- | --- |
| `voxcpm2-bf16.gguf` | identical to the Hugging Face weights | 8.4e-3 decoding, 4.1e-3 encoding: the file stores AudioVAE in bfloat16 |
| `voxcpm2-q8_0.gguf` | prefill 0.9 %, first latent 1.7 % (reference mode 1.0 %), one DiT evaluation 2.7 %, stop logits 0.2 %, the text LM's state by the third patch 10 %: the 8-bit weights | 4.5e-4 decoding (float16 rate embeddings), encoding identical |

The Q8_0 products keep the file's values exact (int8 times the block's
float16 scale, summed in float32) and the activations in bfloat16;
`voxcpm.products` checks them, and the bfloat16 tiles, against a
double-precision sum of the same operands (relative to the sum of the
products' magnitudes, at most 7e-8 for Q8_0 and 1.5e-6 for bfloat16). audio.cpp quantizes the activations of products of two or more
rows to 8 bits as well.

The text tokenizer reproduces 21 of 21 reference cases, among them Chinese
multi-character tokens, byte fallback for emoji and rare CJK, added tokens in
the text and runs of spaces.

## Intelligibility

Gufo's Qwen3-ASR-1.7B transcribed `voxcpm_synthesize` outputs of the
Hugging Face weights:

| Mode | Text | Transcript |
| --- | --- | --- |
| Automatic voice | Hello, this is a first test of VoxCPM two, running natively on Gufo. | Hello. This is a first test of Vox EPM2 running natively on Gufo. |
| Voice design ("A deep male voice, calm and slow") | The weather today is sunny with a light breeze from the west. | The weather today is sunny with a light breeze from the west. |
| Automatic voice, German | Guten Abend! Heute testen wir die Sprachsynthese mit VoxCPM zwei auf dem Strix Halo. | Guten Abend. Heute testen wir die Sprachsynthese mit Vox CPM 2 auf dem Strix Halo. |
| Cloning the designed voice, no transcript | Cloning works from a short recording of the speaker. | Cloning works from a short recording of the speaker. |
| Cloning with its transcript | Cloning works from a short recording of the speaker. | Cloning works from a short recording of the speaker. |
| Automatic voice, 18.6 s | The history of speech synthesis goes back more than two centuries. Early machines imitated the human vocal tract with bellows, reeds and resonators, and later electronic systems shaped sound with filters. Today neural networks learn to speak from thousands of hours of recordings, and a single model can talk in dozens of languages. | The history of speech synthesis goes back more than two centuries. Early machines imitated the human vocal tract with bellows, reeds, and resonators, and later electronic systems shaped sound with filters. Today, neural networks learn to speak from thousands of hours of recordings, and a single model can talk in dozens of languages. |

The 37-word benchmark paragraph ("When the first light reaches the harbor,
speak gently, …"), automatic voice, seed 42, transcribes word for word
with both the Hugging Face weights and `voxcpm2-q8_0.gguf`.

## Gufo's TTS functional suite

`tests/functional/model_services.py tts` checks `gufo serve tts` end to end
(intelligibility through Qwen3-ASR, seeded replay, PCM, WAV and SSE equality,
concurrent requests, rejections, cancellation). Its audio checks expect
24 kHz; with that expectation changed to 48 kHz in a copy of the suite, the
`base` variant (a cloned preset voice from the suite's recording) and the
`design` variant pass all 11 checks, with the Hugging Face weights and with
`voxcpm2-q8_0.gguf` (the copy also lets `--model` name a GGUF file instead
of a directory with `config.json`). Unchanged, the suite fails its first
synthesis on the sample rate.
