# Standalone Qwen3.6/Qwen35MoE Strata adapter.
#
# This project intentionally lives outside Strata's top-level CMake graph.  The
# existing native engine vendors a pinned ggml tree, whereas this adapter is a
# correctness-preserving bridge to a configurable, current llama.cpp checkout
# which already implements Qwen35MoE's hybrid DeltaNet/MoE graph and MTP draft
# path.  Keeping the dependency boundary explicit prevents two incompatible
# ggml copies from ending up in one executable.
cmake_minimum_required(VERSION 3.24)

project(strata_qwen35moe_adapter LANGUAGES C CXX)

set(STRATA_LLAMA_CPP_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party/llama.cpp" CACHE PATH
    "llama.cpp checkout that supplies Qwen35MoE and draft-MTP support")
option(STRATA_QWEN35_ENABLE_CUDA "Build the adapter with llama.cpp CUDA support" ON)

# This host is an x64 i9-12950HX, but CMake's Ninja + MSVC environment can
# leave CMAKE_SYSTEM_PROCESSOR empty.  llama.cpp then falls back to its generic
# CPU backend even with GGML_NATIVE=ON.  Supply the compiler's verified x64
# target only in that narrow case; llama.cpp's own native SIMD probe selects
# AVX/AVX2/FMA/F16C and keeps unsupported AVX-512 disabled.
if(MSVC AND CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "x64")
    if(NOT CMAKE_SYSTEM_PROCESSOR OR CMAKE_SYSTEM_PROCESSOR STREQUAL "UNKNOWN")
        set(CMAKE_SYSTEM_PROCESSOR "AMD64")
    endif()

    # Preserve the verified native outcome in the cache as well.  Without
    # this, llama.cpp's MSVC probe sets non-cache variables while stale cache
    # entries misleadingly keep saying OFF.  This adapter is explicitly for
    # the local i9-12950HX: AVX/AVX2/FMA/F16C/BMI2 are available, AVX-512 is
    # not.  A different architecture never reaches this host policy.
    set(GGML_AVX ON CACHE BOOL "" FORCE)
    set(GGML_AVX2 ON CACHE BOOL "" FORCE)
    set(GGML_FMA ON CACHE BOOL "" FORCE)
    set(GGML_F16C ON CACHE BOOL "" FORCE)
    set(GGML_BMI2 ON CACHE BOOL "" FORCE)
    set(GGML_AVX512 OFF CACHE BOOL "" FORCE)
endif()

if(NOT EXISTS "${STRATA_LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "STRATA_LLAMA_CPP_DIR must name a llama.cpp checkout; got '${STRATA_LLAMA_CPP_DIR}'.\n"
        "For the Windows Qwen3.6 profile use:\n"
        "  -DSTRATA_LLAMA_CPP_DIR=C:/Models/Strata/runtimes/qwen36-35b-a3b/llama.cpp")
endif()

# The prompt-prefill cache restores Qwen MTP's pending target-hidden-state row
# as well as target and draft K/V.  That small, versioned llama.cpp extension
# is deliberately maintained as source-owned patch material because the
# deployed runtime uses an external checkout.  Fail configuration loudly if a
# rebuild is pointed at an unpatched checkout; a silent fallback would leave
# caching disabled despite the production profile advertising it.
set(STRATA_QWEN35_MTP_CACHE_PATCH
    "${CMAKE_CURRENT_LIST_DIR}/patches/llama-mtp-prompt-cache-state.patch")
if(NOT EXISTS "${STRATA_QWEN35_MTP_CACHE_PATCH}")
    message(FATAL_ERROR "Missing required MTP prompt-cache patch: ${STRATA_QWEN35_MTP_CACHE_PATCH}")
endif()
file(READ "${STRATA_LLAMA_CPP_DIR}/common/speculative.cpp" STRATA_QWEN35_SPECULATIVE_SOURCE)
if(NOT STRATA_QWEN35_SPECULATIVE_SOURCE MATCHES "kMtpPromptCacheStateMagic")
    message(FATAL_ERROR
        "The selected llama.cpp checkout lacks Strata's Qwen MTP prompt-cache state patch.\n"
        "Apply ${CMAKE_CURRENT_LIST_DIR}/Apply-LlamaMtpPromptCachePatch.ps1, then reconfigure.")
endif()
find_program(STRATA_QWEN35_GIT_EXECUTABLE git REQUIRED)
execute_process(
    COMMAND "${STRATA_QWEN35_GIT_EXECUTABLE}" -C "${STRATA_LLAMA_CPP_DIR}" rev-parse HEAD
    RESULT_VARIABLE STRATA_QWEN35_GIT_RESULT
    OUTPUT_VARIABLE STRATA_QWEN35_LLAMA_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
set(STRATA_QWEN35_MTP_CACHE_PATCH_BASE "79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6")
if(NOT STRATA_QWEN35_GIT_RESULT EQUAL 0 OR
   NOT STRATA_QWEN35_LLAMA_REVISION STREQUAL STRATA_QWEN35_MTP_CACHE_PATCH_BASE)
    message(FATAL_ERROR
        "The selected llama.cpp checkout is '${STRATA_QWEN35_LLAMA_REVISION}', but the MTP prompt-cache patch is pinned to ${STRATA_QWEN35_MTP_CACHE_PATCH_BASE}.\n"
        "Use that exact checkout revision or rebase and validate the patch before configuring.")
endif()

# We use llama-common's public speculative API.  It is the upstream-tested
# implementation of a Qwen35MoE MTP draft context, rather than an incomplete
# reimplementation in this adapter.
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_SUBPROCESS OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(GGML_NATIVE ON CACHE BOOL "" FORCE)
set(GGML_CUDA ${STRATA_QWEN35_ENABLE_CUDA} CACHE BOOL "" FORCE)
set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
set(GGML_CUDA_FA ON CACHE BOOL "" FORCE)

# llama.cpp places its runtime DLLs and this adapter under <build>/bin.
add_subdirectory("${STRATA_LLAMA_CPP_DIR}" "${CMAKE_BINARY_DIR}/llama.cpp" EXCLUDE_FROM_ALL)

add_executable(strata-qwen35moe-engine
    qwen35moe_engine.cpp
    "${CMAKE_CURRENT_LIST_DIR}/../src/core/qwen35moe_layout.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../src/core/conversation_memory.cpp")

# Keep the adapter beside the DLLs emitted by llama.cpp.  This makes the
# standalone build directly runnable without modifying PATH or copying files
# into a live Strata runtime directory.
set_target_properties(strata-qwen35moe-engine PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

target_compile_features(strata-qwen35moe-engine PRIVATE cxx_std_20)
target_include_directories(strata-qwen35moe-engine PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../include"
    "${STRATA_LLAMA_CPP_DIR}/include"
    "${STRATA_LLAMA_CPP_DIR}/common")
target_link_libraries(strata-qwen35moe-engine PRIVATE llama llama-common)

if(MSVC)
    target_compile_options(strata-qwen35moe-engine PRIVATE /W4 /permissive- /EHsc /utf-8)
else()
    target_compile_options(strata-qwen35moe-engine PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(strata-qwen35moe-limits-test qwen35moe_limits_test.cpp)
    target_compile_features(strata-qwen35moe-limits-test PRIVATE cxx_std_20)
    add_test(NAME strata-qwen35moe-limits COMMAND strata-qwen35moe-limits-test)
    add_executable(strata-qwen35moe-prompt-cache-test qwen35moe_prompt_cache_test.cpp)
    target_compile_features(strata-qwen35moe-prompt-cache-test PRIVATE cxx_std_20)
    add_test(NAME strata-qwen35moe-prompt-cache COMMAND strata-qwen35moe-prompt-cache-test)
    add_test(NAME strata-qwen35moe-cli-help COMMAND strata-qwen35moe-engine --help)
    set_tests_properties(strata-qwen35moe-cli-help PROPERTIES PASS_REGULAR_EXPRESSION "max-output-tokens")
    add_test(NAME strata-qwen35moe-invalid-cache-config
        COMMAND "${CMAKE_COMMAND}"
                "-DENGINE=$<TARGET_FILE:strata-qwen35moe-engine>"
                -P "${CMAKE_CURRENT_LIST_DIR}/qwen35moe_invalid_cache_config_test.cmake")
endif()
