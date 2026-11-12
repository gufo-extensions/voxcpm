# VoxCPM Gufo extension

A community [Gufo](https://github.com/gufo-org/gufo) extension for
[VoxCPM2](https://github.com/OpenBMB/VoxCPM), OpenBMB's tokenizer-free
multilingual text-to-speech model, on AMD Strix Halo (`gfx1151`).

> **Status:** work in progress, not released. Speech synthesis runs through
> `gufo serve tts` with an automatic voice, voice design and voice cloning,
> matches the reference implementation
> ([quality](docs/models/voxcpm2/QUALITY.md)) and is 1.8–1.9 times as fast as
> audio.cpp with the same Q8_0 file, 2.4–2.5 times with bfloat16 weights
> ([benchmarks](docs/models/voxcpm2/BENCHMARKS.md)).

## Model

| Provider | Model | Files |
| --- | --- | --- |
| `voxcpm2` | VoxCPM2 (2B parameters, 30 languages, 48 kHz output) | The Hugging Face directory [`openbmb/VoxCPM2`](https://huggingface.co/openbmb/VoxCPM2), or one of audio.cpp's VoxCPM2 GGUFs |

VoxCPM2 generates speech as continuous latents, one **patch** of four latent
frames (160 ms of audio) at a time. A 28-layer MiniCPM4 language model reads
the text and the patches so far; an 8-layer residual language model adds
acoustic detail; a 12-layer local diffusion transformer turns both into the
next patch in 10 Euler steps with classifier-free guidance; a 12-layer local
encoder feeds the patch back. AudioVAE V2 decodes the latents to 48 kHz audio
and encodes reference recordings at 16 kHz.

The provider loads either layout:

- the Hugging Face directory: `config.json`, `model.safetensors` (bfloat16),
  `audiovae.pth` (float32, read without PyTorch) and `tokenizer.json`;
- one GGUF file of audio.cpp's VoxCPM2 packages
  ([`audio-cpp/audio.cpp-gguf`](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/VoxCPM2-GGUF):
  `voxcpm2-bf16.gguf`, `voxcpm2-orig.gguf`, `voxcpm2-q8_0.gguf`), which embed
  the configuration and tokenizer. Q8_0 matrices stay 8-bit on the GPU: the
  products convert the values exactly and keep the activations in bfloat16
  (prompts of more than 128 rows: see the known limits), so the quantized
  file saves memory and generation time. Float16 and float32 tensors, and
  Q8_0 vectors and embedding tables, are converted to bfloat16 on load.

Other community GGUF conversions of VoxCPM2 use their own layouts and are
not supported.

The weights aren't part of this repository. OpenBMB publishes them under the
Apache-2.0 license.

## Use

```sh
MODEL=/models/VoxCPM2   # the Hugging Face directory, or an audio.cpp GGUF
gufo serve tts --model "$MODEL" \
  --voice anna=/voices/anna.wav --voice-text anna=/voices/anna.txt
```

- `GET /v1/audio/voices` lists `auto` (the model picks a voice),
  `voice-design` and every preset. Requests without `voice` use `auto`.
- **Voice design and style:** `instructions` describes the voice or the
  delivery in natural language ("A young woman, gentle and sweet voice",
  "slightly faster, cheerful tone"); it is put in parentheses before the
  text, as the reference does.
- **Cloning:** a preset (`--voice NAME=WAV`) or `reference_audio`. Without a
  transcript the recording conditions the voice in isolation; with one
  (`--voice-text` or `reference_text`) it is also continued, which the
  reference calls ultimate cloning. Any sample rate and channel count works:
  the recording is mixed down and resampled to 16 kHz. A reference is encoded
  once and cached.
- `seed` selects the diffusion noise; the same seed reproduces the audio
  exactly. `max_new_tokens` caps the patches (6.25 per second); like the
  reference, generation also stops at 6 patches per text token plus 10, and a
  buffered request that reaches that cap is retried with the next seed, up to
  three attempts. Other sampling fields don't apply to VoxCPM2.
- Streams (`response_format: pcm`, SSE) send every 160 ms patch as soon as it
  is decoded.
- Output is 48 kHz mono.

## Known limits

- Speed: generation takes 0.19–0.21 s per second of audio with the Q8_0
  file and 0.31 s with bfloat16 weights on Strix Halo, and the first audio
  arrives after 0.04–0.18 s ([benchmarks](docs/models/voxcpm2/BENCHMARKS.md)).
  That is 1.8–1.9 times as fast as audio.cpp with the same Q8_0 file and
  2.4–2.5 times with bfloat16 weights. The local DiT's products, with their
  22 activation rows, read Q8_0 weights at about 72 % of the memory
  bandwidth ([experiments](docs/models/voxcpm2/EXPERIMENTS.md)).
- Gufo's TTS functional suite expects 24 kHz audio; it passes with that
  expectation changed to 48 kHz (see [quality](docs/models/voxcpm2/QUALITY.md)).
- One request runs at a time; Gufo queues the others.
- `gufo serve tts` limits the GPU to one hardware queue for speech models;
  set `GPU_MAX_HW_QUEUES=2` before starting it so that AudioVAE decodes
  alongside generation (3–4 % faster,
  [benchmarks](docs/models/voxcpm2/BENCHMARKS.md)).
- The decoding parameters are the reference's defaults (10 Euler steps,
  guidance 2.0); Gufo's API has no fields for them.
- Prompts of more than 128 rows (text tokens and reference patches) run on
  hipBLASLt, which reads bfloat16: with the Q8_0 file, each matrix of their
  prefill is first expanded to bfloat16 (rounded to 8 significant bits), so
  their prefill computes with slightly different weights than generation.
- Not implemented: the reference's optional text normalization and
  denoiser, LoRA adapters, VoxCPM 0.5B and 1.5.
- The tokenizer follows `tokenizer.json` as transformers 4 applies it.
  transformers 5 tokenizes text that starts with a space differently;
  the IDs differ only there.

## Responsible use

Voice cloning produces realistic speech in another person's voice. Clone
only voices you are allowed to use, and don't use the extension for
impersonation, fraud or disinformation; OpenBMB forbids those uses of VoxCPM
and recommends marking generated audio as such
([risks and limitations](https://github.com/OpenBMB/VoxCPM/blob/300e4260b509066151d623dc9c0aa922cf82232b/README.md#%EF%B8%8F-risks-and-limitations)).
The extension adds no watermark or other marking to the audio; laws on
synthetic media, such as labeling duties, may apply where you publish it.

## Build with Gufo

The extension needs extension API 0.3.0 (Gufo `main`); `flake.lock` pins the
revision it was built and tested with.

```sh
git clone https://github.com/gufo-org/gufo
cd gufo
cmake --preset release -DGUFO_EXTENSION_DIRS=/path/to/voxcpm
cmake --build --preset release --parallel 4
./build/release/gufo diagnose --section extensions
```

With Nix, `nix build` in this repository builds the pinned Gufo revision with
this extension.

## Tests

CPU tests (configuration, probe, tokenizer, weight readers) run without the
model, and so does the GPU test of the matrix-core products and weight tiles
(`voxcpm.products`, against a double-precision sum of the same operands).
The model tests read `GUFO_VOXCPM_MODEL` (the Hugging Face directory) and
`GUFO_VOXCPM_GGUF` (an audio.cpp GGUF) and are skipped without them; they
compare the generator and the AudioVAE with tensors that
`tools/reference/generator_golden.py` recorded from OpenBMB's
implementation.

`voxcpm_synthesize MODEL OUT.wav TEXT [--instruct TEXT] [--reference AUDIO
[--reference-text TEXT]] [--seed N] [--cfg G] [--timesteps N] [--repeat N]
[--stream]` is the development driver; `voxcpm_gemm_bench [REPEATS [q8]]`
times the product kernels at the model's shapes.

## License

The extension source is MIT licensed ([LICENSE](LICENSE)). Code adapted from
other projects keeps their licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
