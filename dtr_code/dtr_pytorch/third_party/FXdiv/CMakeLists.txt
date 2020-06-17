CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12 FATAL_ERROR)

INCLUDE(GNUInstallDirs)

# ---[ Project
PROJECT(FXdiv C CXX)

# ---[ Options.
OPTION(FXDIV_BUILD_TESTS "Build FXdiv unit tests" ON)
OPTION(FXDIV_BUILD_BENCHMARKS "Build FXdiv micro-benchmarks" ON)
OPTION(FXDIV_USE_INLINE_ASSEMBLY "Allow use of inline assembly in FXdiv" OFF)

# ---[ CMake options
IF(FXDIV_BUILD_TESTS)
  ENABLE_TESTING()
ENDIF()

# ---[ Download deps
SET(CONFU_DEPENDENCIES_SOURCE_DIR ${CMAKE_SOURCE_DIR}/deps
  CACHE PATH "Confu-style dependencies source directory")
SET(CONFU_DEPENDENCIES_BINARY_DIR ${CMAKE_BINARY_DIR}/deps
  CACHE PATH "Confu-style dependencies binary directory")

IF(FXDIV_BUILD_TESTS AND NOT DEFINED GOOGLETEST_SOURCE_DIR)
  MESSAGE(STATUS "Downloading Google Test to ${CONFU_DEPENDENCIES_SOURCE_DIR}/googletest (define GOOGLETEST_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadGoogleTest.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download")
  SET(GOOGLETEST_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/googletest" CACHE STRING "Google Test source directory")
ENDIF()

IF(FXDIV_BUILD_BENCHMARKS AND NOT DEFINED GOOGLEBENCHMARK_SOURCE_DIR)
  MESSAGE(STATUS "Downloading Google Benchmark to ${CONFU_DEPENDENCIES_SOURCE_DIR}/googlebenchmark (define GOOGLEBENCHMARK_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadGoogleBenchmark.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/googlebenchmark-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googlebenchmark-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googlebenchmark-download")
  SET(GOOGLEBENCHMARK_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/googlebenchmark" CACHE STRING "Google Benchmark source directory")
ENDIF()

# ---[ FXdiv library
IF(${CMAKE_VERSION} VERSION_LESS "3.0")
  ADD_LIBRARY(fxdiv STATIC include/fxdiv.h)
  SET_TARGET_PROPERTIES(fxdiv PROPERTIES LINKER_LANGUAGE C)
ELSE()
  ADD_LIBRARY(fxdiv INTERFACE)
ENDIF()
TARGET_INCLUDE_DIRECTORIES(fxdiv INTERFACE include)
IF(NOT FXDIV_USE_INLINE_ASSEMBLY)
  TARGET_COMPILE_DEFINITIONS(fxdiv INTERFACE FXDIV_USE_INLINE_ASSEMBLY=0)
ENDIF()

INSTALL(FILES include/fxdiv.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

IF(FXDIV_BUILD_TESTS)
  # ---[ Build google test
  IF(NOT TARGET gtest)
    SET(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    ADD_SUBDIRECTORY(
      "${GOOGLETEST_SOURCE_DIR}"
      "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest")
  ENDIF()

  ADD_EXECUTABLE(multiply-high-test test/multiply-high.cc)
  TARGET_LINK_LIBRARIES(multiply-high-test fxdiv gtest gtest_main)
  ADD_TEST(multiply-high multiply-high-test)

  ADD_EXECUTABLE(quotient-test test/quotient.cc)
  TARGET_LINK_LIBRARIES(quotient-test fxdiv gtest gtest_main)
  ADD_TEST(quotient quotient-test)
ENDIF()

IF(FXDIV_BUILD_BENCHMARKS)
  # ---[ Build google benchmark
  IF(NOT TARGET benchmark)
    SET(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    ADD_SUBDIRECTORY(
      "${GOOGLEBENCHMARK_SOURCE_DIR}"
      "${CONFU_DEPENDENCIES_BINARY_DIR}/googlebenchmark")
  ENDIF()

  ADD_EXECUTABLE(init-bench bench/init.cc)
  TARGET_LINK_LIBRARIES(init-bench fxdiv benchmark)

  ADD_EXECUTABLE(multiply-bench bench/multiply.cc)
  TARGET_LINK_LIBRARIES(multiply-bench fxdiv benchmark)

  ADD_EXECUTABLE(divide-bench bench/divide.cc)
  TARGET_LINK_LIBRARIES(divide-bench fxdiv benchmark)

  ADD_EXECUTABLE(quotient-bench bench/quotient.cc)
  TARGET_LINK_LIBRARIES(quotient-bench fxdiv benchmark)

  ADD_EXECUTABLE(round-down-bench bench/round-down.cc)
  TARGET_LINK_LIBRARIES(round-down-bench fxdiv benchmark)
ENDIF()
