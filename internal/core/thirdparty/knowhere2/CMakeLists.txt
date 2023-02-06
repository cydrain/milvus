#-------------------------------------------------------------------------------
# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License.
#-------------------------------------------------------------------------------

set( KNOWHERE_VERSION v1.3.6 )
set( KNOWHERE_SOURCE_MD5 "e711e204a3a6c4918352d464c0b87793")

if ( DEFINED ENV{MILVUS_KNOWHERE_URL} )
    set( KNOWHERE_SOURCE_URL "$ENV{MILVUS_KNOWHERE_URL}" )
else ()
    set( KNOWHERE_SOURCE_URL
        "https://github.com/milvus-io/knowhere/archive/refs/tags/${KNOWHERE_VERSION}.tar.gz" )
endif ()

message(STATUS "Building knowhere-${KNOWHERE_VERSION} from source")
message(STATUS ${CMAKE_BUILD_TYPE})

if ( BUILD_DISK_ANN STREQUAL "ON" )
    set(KNOWHERE_WITH_DISKANN ON CACHE BOOL "" FORCE )
else ()
    set(KNOWHERE_WITH_DISKANN OFF CACHE BOOL "" FORCE )
endif ()

# BUILD_UNIT_TEST OFF will only affect knowhere
set( BUILD_UNIT_TEST  OFF )
FetchContent_Declare(
        knowhere
        GIT_REPOSITORY  "https://github.com/cydrain/knowhere.git"
        GIT_TAG         caiyd_support_prometheus_metric
        SOURCE_DIR      ${CMAKE_CURRENT_BINARY_DIR}/knowhere-src
        BINARY_DIR      ${CMAKE_CURRENT_BINARY_DIR}/knowhere-build
        DOWNLOAD_DIR    ${THIRDPARTY_DOWNLOAD_PATH} )

FetchContent_GetProperties( knowhere )
if ( NOT knowhere_POPULATED )
    FetchContent_Populate( knowhere )

    # Adding the following target:
    # knowhere
    add_subdirectory( ${knowhere_SOURCE_DIR}
                      ${knowhere_BINARY_DIR} )
endif()

# get prometheus COMPILE_OPTIONS
get_property( var DIRECTORY "${knowhere_SOURCE_DIR}" PROPERTY COMPILE_OPTIONS )
message( STATUS "knowhere src compile options: ${var}" )
