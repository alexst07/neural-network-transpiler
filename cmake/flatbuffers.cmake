include(ExternalProject)

set(flatbuffers_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/external/flatbuffers)

ExternalProject_Add(
    flatbuffers
    PREFIX ${flatbuffers_PREFIX}
    URL "https://github.com/google/flatbuffers/archive/v1.7.1.tar.gz"
    URL_MD5 "81934736f31fbd2cfdb513e71b53b358"
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${flatbuffers_PREFIX} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    LOG_UPDATE ON
    LOG_CONFIGURE ON
    LOG_BUILD ON
)

set(FLATBUFFERS_INCLUDE_DIRS "${flatbuffers_PREFIX}/include")
set(FLATBUFFERS_FLATC_EXECUTABLE "${flatbuffers_PREFIX}/bin/flatc")
set(FLATBUFFERS_LIBRARIES "${flatbuffers_PREFIX}/lib/libflatbuffers.a")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildFlatBuffers.cmake)
