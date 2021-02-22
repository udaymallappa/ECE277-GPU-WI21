include_directories(
  ${SOURCE_DIR}
  ${CUDA_TOOLKIT_INCLUDE}
  ${NVTX_INCLUDE_DIR}
  ${NVML_INCLUDE_DIR}
  ${FREEGLUT_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR}
  ${CUDA_TOOLKIT_SAMPLES_DIR}/common/inc
)

#######################################
#######################################
file(GLOB HEADERS *.h *.cuh)
source_group("Headers" FILES ${HEADERS})


cuda_compile_ptx(c1_opcode_ptx
opcode.cu
OPTIONS --cudart=shared -lineinfo -arch sm_52 -use_fast_math -maxrregcount 104
DEBUG -Xptxas -v -g -G
)

#cuda_compile_cubin(opcode_cubin
#opcode.cu
#OPTIONS --cudart=shared -lineinfo -arch sm_52 -use_fast_math -maxrregcount 104
#DEBUG -Xptxas -v -g -G
#)


cuda_add_executable( c1_opcode
${HEADERS}
opcode.cu
OPTIONS --cudart=shared -lineinfo -arch sm_52 -use_fast_math -maxrregcount 104
DEBUG -Xptxas -v -g -G 
)

target_link_libraries( 
c1_opcode

${CUDA_CUDA_LIBRARY}
${CUDA_CUDART_LIBRARY}
${CUDA_CURAND_LIBRARY}
${NVML_DEBUG_LIBRARIES}
${NVTX_DEBUG_LIBRARIES}
${FREEGLUT_DEBUG_LIBRARIES}
${GLEW_DEBUG_LIBRARIES}
)

target_link_libraries( 
c1_opcode
${CUDA_CUDA_LIBRARY}
${CUDA_CUDART_LIBRARY}
${CUDA_CURAND_LIBRARY}
${FREEGLUT_RELEASE_LIBRARIES}
${GLEW_RELEASE_LIBRARIES}
)
