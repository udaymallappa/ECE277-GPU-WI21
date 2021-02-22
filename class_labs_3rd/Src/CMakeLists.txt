set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

add_subdirectory(c1_hello)
add_subdirectory(c1_opcode)
add_subdirectory(c1_bandwidthTest)
add_subdirectory(c1_block1d2d)

add_subdirectory(c2_nsight_dbg)
add_subdirectory(c2_measure_exectime)
add_subdirectory(c2_inline_asm)

add_subdirectory(c3_global_mem)
add_subdirectory(c3_aos)
add_subdirectory(c3_fp16)
add_subdirectory(c3_random)
add_subdirectory(c3_matrix_transpose)

add_subdirectory(c4_shared_mem)