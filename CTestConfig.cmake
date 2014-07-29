## This file should be placed in the root directory of your project.
## Then modify the CMakeLists.txt file in the root directory of your
## project to incorporate the testing dashboard.
## # The following are required to uses Dart and the Cdash dashboard
##   ENABLE_TESTING()
##   INCLUDE(CTest)

set(CTEST_PROJECT_NAME "ChestImagingPlatform")
set(CTEST_NIGHTLY_START_TIME "00:00:00 EST")

set(CTEST_DROP_METHOD "http")
set(CTEST_DROP_SITE "ec2-54-187-247-17.us-west-2.compute.amazonaws.com")
set(CTEST_DROP_LOCATION "/CDash/submit.php?project=ChestImagingPlatform")
set(CTEST_DROP_SITE_CDASH TRUE)
