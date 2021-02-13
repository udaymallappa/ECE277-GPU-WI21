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

cuda_add_library (qagent
agent.cu
OPTIONS --cudart=static  -lineinfo -arch sm_52 -use_fast_math
DEBUG -Xptxas -v -g -G 
)

link_libraries( 
qagent

${CUDA_CUDA_LIBRARY}
${CUDA_CUDART_LIBRARY}
${CUDA_CURAND_LIBRARY}
${NVML_DEBUG_LIBRARIES}
${NVTX_DEBUG_LIBRARIES}
${FREEGLUT_DEBUG_LIBRARIES}
${GLEW_DEBUG_LIBRARIES}
)

link_libraries( 
qagent

${CUDA_CUDA_LIBRARY}
${CUDA_CUDART_LIBRARY}
${CUDA_CURAND_LIBRARY}
${FREEGLUT_RELEASE_LIBRARIES}
${GLEW_RELEASE_LIBRARIES}
)

add_custom_command(TARGET qagent POST_BUILD     
COMMAND ${CMAKE_COMMAND} -E copy_if_different 
"${CUDA_TOOLKIT_SAMPLES_DIR}/bin/win64/Release/freeglut.dll"  
$<TARGET_FILE_DIR:qagent>)          

add_custom_command(TARGET qagent POST_BUILD     
COMMAND ${CMAKE_COMMAND} -E copy_if_different 
"${CUDA_TOOLKIT_SAMPLES_DIR}/bin/win64/Release/glew64.dll"  
$<TARGET_FILE_DIR:qagent>)   