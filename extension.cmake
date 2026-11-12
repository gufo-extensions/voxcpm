# VoxCPM (OpenBMB) text-to-speech extension.

# Host side: configuration, probe, tokenizer and the registration.
gufo_extension_target(
  NAME gufo_voxcpm
  SOURCES
    config.cpp
    safetensors.cpp
    torch_checkpoint.cpp
    model_files.cpp
    tokenizer.cpp
    probe.cpp
    extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
)
if(TARGET gufo_voxcpm)
  target_include_directories(gufo_voxcpm PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
  if(ENGINE_ENABLE_HIP)
    target_compile_definitions(gufo_voxcpm PRIVATE ENGINE_ENABLE_HIP=1)
  endif()
endif()

# Device side: the generator's kernels and runtime.
gufo_extension_target(
  NAME gufo_voxcpm_hip
  SOURCES
    gemm.cpp
    weights.cpp
    transformer.cpp
    generator.cpp
    vae.cpp
    synthesizer.cpp
  LINK_LIBRARIES
    gufo_voxcpm
    gufo_ext_runtime
    gufo_ext_hip
    hip::host
    roc::hipblas
    roc::hipblaslt
)
if(TARGET gufo_voxcpm_hip)
  gufo_extension_hip_sources(gufo_voxcpm_hip
    SOURCES
      kernels/generator_kernels.hip
      kernels/vae_kernels.hip
      kernels/skinny_gemm.hip
      kernels/wmma_products.hip
      kernels/convert_kernels.hip
    OPTIONS
      -O3 -Wno-unused-command-line-argument
  )
  target_include_directories(gufo_voxcpm_hip PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
endif()

# Speech synthesis provider: the TtsRunner and its registration.
gufo_extension_target(
  NAME gufo_voxcpm_tts
  SOURCES
    runner.cpp
    provider.cpp
  LINK_LIBRARIES
    gufo_voxcpm_hip
    gufo_voxcpm
    gufo_ext_runtime
    gufo_ext_hip
)

# gufo_ext_runtime only declares the public services; these link their
# implementations into executables that stand apart from `gufo`.
set(_gufo_voxcpm_test_libs gufo_ext_core gufo_ext_cache)
if(ENGINE_ENABLE_HIP)
  list(APPEND _gufo_voxcpm_test_libs gufo_ext_hip)
endif()
set(_gufo_voxcpm_fixtures "${GUFO_EXTENSION_BUILD_ROOT}/tests/fixtures")

gufo_extension_target(
  NAME voxcpm_config_test
  SOURCES
    tests/config_test.cpp
  LINK_LIBRARIES
    gufo_voxcpm
    gufo_ext_runtime
    ${_gufo_voxcpm_test_libs}
)
if(TARGET voxcpm_config_test)
  add_test(NAME voxcpm.config COMMAND voxcpm_config_test)
  set_tests_properties(voxcpm.config PROPERTIES LABELS "voxcpm;cpu")
endif()

gufo_extension_target(
  NAME voxcpm_probe_test
  SOURCES
    tests/probe_test.cpp
  LINK_LIBRARIES
    gufo_voxcpm
    gufo_ext_runtime
    ${_gufo_voxcpm_test_libs}
)
if(TARGET voxcpm_probe_test)
  add_test(NAME voxcpm.probe COMMAND voxcpm_probe_test samples)
  set_tests_properties(voxcpm.probe PROPERTIES LABELS "voxcpm;cpu")
  add_test(NAME voxcpm.probe_model COMMAND voxcpm_probe_test model)
  set_tests_properties(voxcpm.probe_model PROPERTIES
    LABELS "voxcpm;cpu;external-model"
    SKIP_RETURN_CODE 77)
endif()

gufo_extension_target(
  NAME voxcpm_tokenizer_test
  SOURCES
    tests/tokenizer_test.cpp
  LINK_LIBRARIES
    gufo_voxcpm
    gufo_ext_runtime
    ${_gufo_voxcpm_test_libs}
)
if(TARGET voxcpm_tokenizer_test)
  target_compile_definitions(voxcpm_tokenizer_test PRIVATE
    GUFO_VOXCPM_TOKENIZER_GOLDEN="${_gufo_voxcpm_fixtures}/tokenizer_golden.json")
  add_test(NAME voxcpm.tokenizer COMMAND voxcpm_tokenizer_test)
  set_tests_properties(voxcpm.tokenizer PROPERTIES
    LABELS "voxcpm;cpu;tokenizer;external-model"
    SKIP_RETURN_CODE 77)
endif()

gufo_extension_target(
  NAME voxcpm_weights_test
  SOURCES
    tests/weights_test.cpp
  LINK_LIBRARIES
    gufo_voxcpm
    gufo_ext_runtime
    ${_gufo_voxcpm_test_libs}
)
if(TARGET voxcpm_weights_test)
  target_compile_definitions(voxcpm_weights_test PRIVATE
    GUFO_VOXCPM_FIXTURES="${_gufo_voxcpm_fixtures}")
  add_test(NAME voxcpm.weights COMMAND voxcpm_weights_test samples)
  set_tests_properties(voxcpm.weights PROPERTIES LABELS "voxcpm;cpu")
  add_test(NAME voxcpm.weights_model COMMAND voxcpm_weights_test model)
  set_tests_properties(voxcpm.weights_model PROPERTIES
    LABELS "voxcpm;cpu;external-model"
    SKIP_RETURN_CODE 77)
endif()

gufo_extension_target(
  NAME voxcpm_generator_test
  SOURCES
    tests/generator_test.cpp
  LINK_LIBRARIES
    gufo_voxcpm_hip
    gufo_voxcpm
    gufo_ext_runtime
    ${_gufo_voxcpm_test_libs}
)
if(TARGET voxcpm_generator_test)
  target_compile_definitions(voxcpm_generator_test PRIVATE
    GUFO_VOXCPM_FIXTURES="${_gufo_voxcpm_fixtures}")
  add_test(NAME voxcpm.generator COMMAND voxcpm_generator_test)
  set_tests_properties(voxcpm.generator PROPERTIES
    LABELS "voxcpm;gpu;external-model"
    SKIP_RETURN_CODE 77)
  add_test(NAME voxcpm.generator_gguf COMMAND voxcpm_generator_test gguf)
  set_tests_properties(voxcpm.generator_gguf PROPERTIES
    LABELS "voxcpm;gpu;external-model"
    SKIP_RETURN_CODE 77)
endif()

gufo_extension_target(
  NAME voxcpm_vae_test
  SOURCES
    tests/vae_test.cpp
  LINK_LIBRARIES
    gufo_voxcpm_hip
    gufo_voxcpm
    gufo_ext_runtime
    ${_gufo_voxcpm_test_libs}
)
if(TARGET voxcpm_vae_test)
  target_compile_definitions(voxcpm_vae_test PRIVATE
    GUFO_VOXCPM_FIXTURES="${_gufo_voxcpm_fixtures}")
  add_test(NAME voxcpm.vae COMMAND voxcpm_vae_test)
  set_tests_properties(voxcpm.vae PROPERTIES
    LABELS "voxcpm;gpu;external-model"
    SKIP_RETURN_CODE 77)
  add_test(NAME voxcpm.vae_gguf COMMAND voxcpm_vae_test gguf)
  set_tests_properties(voxcpm.vae_gguf PROPERTIES
    LABELS "voxcpm;gpu;external-model"
    SKIP_RETURN_CODE 77)
endif()

gufo_extension_target(
  NAME voxcpm_products_test
  SOURCES
    tests/products_test.cpp
  LINK_LIBRARIES
    gufo_voxcpm_hip
    gufo_voxcpm
    gufo_ext_runtime
    ${_gufo_voxcpm_test_libs}
)
if(TARGET voxcpm_products_test)
  add_test(NAME voxcpm.products COMMAND voxcpm_products_test)
  set_tests_properties(voxcpm.products PROPERTIES
    LABELS "voxcpm;gpu"
    SKIP_RETURN_CODE 77)
endif()

gufo_extension_target(
  NAME voxcpm_synthesize
  SOURCES
    tools/synthesize/synthesize.cpp
  LINK_LIBRARIES
    gufo_voxcpm_hip
    gufo_voxcpm
    gufo_ext_runtime
    ${_gufo_voxcpm_test_libs}
)

gufo_extension_target(
  NAME voxcpm_gemm_bench
  SOURCES
    tools/bench/gemm_bench.cpp
  LINK_LIBRARIES
    gufo_voxcpm_hip
    gufo_voxcpm
    gufo_ext_runtime
    ${_gufo_voxcpm_test_libs}
)
