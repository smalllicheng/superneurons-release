##########################################################
# find and set glog

if (NOT GLOG_ROOT)
    message(FATAL_ERROR "Please specify glog root path in config file.")
endif ()

include_directories(${GLOG_ROOT}/include)
include_directories(${ZFP_ROOT_DIR}/include)
if (APPLE)
    set(GLOG_LIBRARIES ${GLOG_ROOT}/lib/libglog.dylib)
else()
    set(GLOG_LIBRARIES ${GLOG_ROOT}/lib/libglog.so)
endif ()

set(ZFP_LIBRARIES ${ZFP_ROOT_DIR}/build/lib/libzfp.so)

list(APPEND THIRD_LIBS ${GLOG_LIBRARIES})
list(APPEND THIRD_LIBS ${ZFP_LIBRARIES})
