# Third-party notices

This extension contains code adapted from the projects below. Their license
texts are in [licenses/](licenses/) and are installed with the extension.

| Project | Used for | License | Revision | Source |
| --- | --- | --- | --- | --- |
| Gufo | Safetensors reader (`safetensors.cpp`), adapted from Qwen3-TTS; the hipBLASLt plan cache (`gemm.cpp`), adapted from Qwen3-ASR (both by way of the OmniVoice extension) | `MIT` | `5f9612c3b19ab6676b842095bc88bb9b32da5cf0` | [gufo-org/gufo](https://github.com/gufo-org/gufo) |
| VoxCPM | The generator (`generator.cpp`, `transformer.cpp`, `kernels/generator_kernels.hip`) ports VoxCPM2Model with its MiniCPM4 stacks, local encoder, local DiT and CFM sampler; `vae.cpp` and `kernels/vae_kernels.hip` port AudioVAE V2; `synthesizer.cpp` follows its prompt construction, stop and retry rules; `tokenizer.cpp` applies its multi-character Chinese token split | `Apache-2.0` | `300e4260b509066151d623dc9c0aa922cf82232b` | [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) |
| DAC | Through VoxCPM's AudioVAE V2, which VoxCPM bases on it: the convolution blocks and the Snake activation that `vae.cpp` and `kernels/vae_kernels.hip` port | `MIT` | `c7cfc5d2647e26471dc394f95846a0830e7bec34` | [descriptinc/descript-audio-codec](https://github.com/descriptinc/descript-audio-codec) |
| CosyVoice | Through VoxCPM's local DiT, which VoxCPM credits to CosyVoice's flow matching (`generator.cpp`, `kernels/generator_kernels.hip`) | `Apache-2.0` | `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc` | [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) |

| License text | Projects |
| --- | --- |
| [licenses/gufo.txt](licenses/gufo.txt) | Gufo |
| [licenses/VoxCPM.txt](licenses/VoxCPM.txt) | VoxCPM |
| [licenses/DAC.txt](licenses/DAC.txt) | DAC |
| [licenses/CosyVoice.txt](licenses/CosyVoice.txt) | CosyVoice |

The test fixtures in `tests/fixtures/` hold token IDs and float32 tensors
(hidden states, latents, noise, waveform samples) that `tools/reference/`
computed with VoxCPM's reference implementation and the released
checkpoint, and two small `torch.save` files of made-up values;
`tests/config_samples.hpp` holds the checkpoint's `config.json`. None of them
contains weights.
