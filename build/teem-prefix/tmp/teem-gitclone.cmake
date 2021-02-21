
if(NOT "/home/mylap9/CIP_Feb2021/ChestImagingPlatform/build/teem-prefix/src/teem-stamp/teem-gitinfo.txt" IS_NEWER_THAN "/home/mylap9/CIP_Feb2021/ChestImagingPlatform/build/teem-prefix/src/teem-stamp/teem-gitclone-lastrun.txt")
  message(STATUS "Avoiding repeated git clone, stamp file is up to date: '/home/mylap9/CIP_Feb2021/ChestImagingPlatform/build/teem-prefix/src/teem-stamp/teem-gitclone-lastrun.txt'")
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "/home/mylap9/CIP_Feb2021/ChestImagingPlatform/build/teem"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/home/mylap9/CIP_Feb2021/ChestImagingPlatform/build/teem'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git"  clone --no-checkout "git://github.com/Slicer/teem" "teem"
    WORKING_DIRECTORY "/home/mylap9/CIP_Feb2021/ChestImagingPlatform/build"
    RESULT_VARIABLE error_code
    )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once:
          ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'git://github.com/Slicer/teem'")
endif()

execute_process(
  COMMAND "/usr/bin/git"  checkout e4746083c0e1dc0c137124c41eca5d23adf73bfa --
  WORKING_DIRECTORY "/home/mylap9/CIP_Feb2021/ChestImagingPlatform/build/teem"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'e4746083c0e1dc0c137124c41eca5d23adf73bfa'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/bin/git"  submodule update --recursive --init 
    WORKING_DIRECTORY "/home/mylap9/CIP_Feb2021/ChestImagingPlatform/build/teem"
    RESULT_VARIABLE error_code
    )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/home/mylap9/CIP_Feb2021/ChestImagingPlatform/build/teem'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy
    "/home/mylap9/CIP_Feb2021/ChestImagingPlatform/build/teem-prefix/src/teem-stamp/teem-gitinfo.txt"
    "/home/mylap9/CIP_Feb2021/ChestImagingPlatform/build/teem-prefix/src/teem-stamp/teem-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/home/mylap9/CIP_Feb2021/ChestImagingPlatform/build/teem-prefix/src/teem-stamp/teem-gitclone-lastrun.txt'")
endif()

