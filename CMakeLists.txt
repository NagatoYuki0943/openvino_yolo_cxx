cmake_minimum_required (VERSION 3.10)
project("openvino_yolo11_cxx")

set(CMAKE_CXX_STANDARD 20)
# 设置为 ON 表示指定的 C++ 标准是必需的
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 设置为 OFF 表示禁用编译器特定的语言扩展
# 具体来说：
#   当设置为 OFF 时，编译器会使用纯标准的 C++ 标志，例如 -std=c++17
#   当设置为 ON（默认值）时，编译器会使用带扩展的标志，例如 -std=gnu++17
# 禁用扩展可以确保你的代码具有更好的可移植性，能够在不同的编译器之间兼容
set(CMAKE_CXX_EXTENSIONS OFF)

# 如果支持，请为 MSVC 编译器启用热重载。
if (POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

message("CMAKE_PREFIX_PATH: " ${CMAKE_PREFIX_PATH})

# opencv
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
    message("OpenCV_LIBS: " ${OpenCV_LIBS})
endif()

# openvino
find_package(OpenVINO REQUIRED)
if(OpenVINO_FOUND)
    # 获取目标接口的包含目录
    get_target_property(OV_INCLUDE_DIR openvino::runtime INTERFACE_INCLUDE_DIRECTORIES)
    message("OpenVINO Target Include Dirs: ${OV_INCLUDE_DIR}")
endif()

# xtensor
# find_package(xtensor REQUIRED)
# if(xtensor_FOUND)
#     message("xtensor_INCLUDE_DIRS: " ${xtensor_INCLUDE_DIRS})
# endif()

# mkl
# find_package(MKL REQUIRED)
# if(MKL_FOUND)
#     message("MKL_INCLUDE_DIRS: " ${MKL_INCLUDE})
#     message("MKL_LIBRARIES: " MKL::MKL)
# endif()

# openblas
# find_package(OpenBLAS REQUIRED)

# xtensor-blas
# find_package(xtensor-blas REQUIRED)
# if(xtensor-blas_FOUND)
#     message("xtensor-blas_INCLUDE_DIRS: " ${xtensor_blas_INCLUDE_DIRS})
# endif()

# 将源代码添加到此项目的可执行文件。
set(SOURCES
    main
    "main.cpp"
    "src/yolo/openvino_yolo11_det_inference.hpp"
)

# 添加可执行文件
add_executable(main ${SOURCES})

# 添加头文件搜索路径
# 只需要添加你自己项目内部的 include 文件夹
target_include_directories(main PRIVATE include)

# 添加链接目录
# 现代 CMake 推荐：只要 link 了目标，头文件和编译选项会自动传递
target_link_libraries(main PRIVATE
    ${OpenCV_LIBS}
    openvino::runtime
    # MKL::MKL
    # xtensor
    # xtensor-blas
)

# 添加编译选项
target_compile_definitions(main PRIVATE
    HAVE_CMKL=1
    # HAVE_CBLAS=1
    # xtensor 使用 xsimd 加速(windows(x64) 有 bug, linux(x64/arm) 可以用)
    # XTENSOR_USE_XSIMD
)

# 添加编译器选项
target_compile_options(main PRIVATE
    $<$<AND:$<PLATFORM_ID:Windows>,$<CXX_COMPILER_ID:MSVC>>:/bigobj>
    $<$<AND:$<PLATFORM_ID:Windows>,$<CXX_COMPILER_ID:MSVC>>:/utf-8>
    $<$<AND:$<PLATFORM_ID:Windows>,$<CXX_COMPILER_ID:GNU>>:-Wa,-mbig-obj>
    # 优化
    $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:/O2> # MSVC: O1:最小化体积 /O2:最大化速度 /Od: 关闭优化
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-O3> # 其他编译器(GCC/Clang): Os: 优化大小 O3: 优化速度
    # 指令集优化
    # $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:/arch:AVX> # 为 MSVC 启用 AVX
    $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:/arch:AVX2> # 为 MSVC 启用 AVX2
    # $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:/arch:AVX512> # 为 MSVC 启用 AVX512
    $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-march=native> # 其他编译器(GCC/Clang): -march=native: 自动选择最适合本机的指令集
    # 允许 MSVC 对标量代码进行 FMA 优化
    $<$<CXX_COMPILER_ID:MSVC>:/fp:fast>
    # 使用链接器垃圾回收
    # 为 MSVC 设置 /Gy
    $<$<CXX_COMPILER_ID:MSVC>:/Gy>
    # 为 GCC/Clang 设置 -ffunction-sections 和 -fdata-sections
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-ffunction-sections -fdata-sections>
)

# 链接选项:
target_link_options(main PRIVATE
    # 为 MSVC 设置 /OPT:REF
    $<$<CXX_COMPILER_ID:MSVC>:/OPT:REF>
    # 为 GCC/Clang 设置 -Wl,--gc-sections
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wl,--gc-sections>
)

# 只在非 MSVC 编译器且 CMAKE_STRIP 可用时，才添加 strip 命令
if(NOT MSVC AND CMAKE_STRIP)
    add_custom_command(
        TARGET main POST_BUILD
        COMMAND ${CMAKE_STRIP} $<TARGET_FILE:main>
        COMMENT "Stripping symbols from $<TARGET_FILE:main>"
    )
endif()
