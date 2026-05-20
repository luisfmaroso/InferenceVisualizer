# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "")
  file(REMOVE_RECURSE
  "src\\CMakeFiles\\InferenceVisualizer_autogen.dir\\AutogenUsed.txt"
  "src\\CMakeFiles\\InferenceVisualizer_autogen.dir\\ParseCache.txt"
  "src\\InferenceVisualizer_autogen"
  )
endif()
