# Verifies parser rejection without loading a model.  CTest's WILL_FAIL and
# PASS_REGULAR_EXPRESSION properties do not compose for a non-zero process, so
# keep the expected failure inside this small test driver.
if(NOT DEFINED ENGINE OR ENGINE STREQUAL "")
    message(FATAL_ERROR "ENGINE is required")
endif()
execute_process(
    COMMAND "${ENGINE}" --serve --model placeholder.gguf
            --conversation-cache-mib 1 --conversation-cache-slots 0
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)
set(combined "${stdout}\n${stderr}")
if(result EQUAL 0)
    message(FATAL_ERROR "invalid cache configuration unexpectedly succeeded")
endif()
if(NOT combined MATCHES "positive --conversation-cache-slots")
    message(FATAL_ERROR "invalid cache configuration emitted the wrong diagnostic: ${combined}")
endif()
