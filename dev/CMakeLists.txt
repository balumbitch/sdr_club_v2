cmake_minimum_required(VERSION 3.16)
project(PlutoSDR LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

# Опции
option(BUILD_TX "Build transmitter" ON)
option(BUILD_RX "Build receiver with GUI" ON)
option(BUILD_MAIN "Build original combined version" OFF)

# Пути
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)
set(DEV_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Сообщения о путях
message(STATUS "Project source dir: ${PROJECT_SOURCE_DIR}")
message(STATUS "Current source dir: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Third party dir: ${THIRD_PARTY_DIR}")
message(STATUS "Dev dir: ${DEV_DIR}")

# Поиск пакетов
find_package(SoapySDR REQUIRED)
find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

# Поиск FFTW3
pkg_check_modules(FFTW3 REQUIRED fftw3)
if(FFTW3_FOUND)
    message(STATUS "FFTW3 found: ${FFTW3_LIBRARIES}")
else()
    message(FATAL_ERROR "FFTW3 not found! Install with: sudo apt install libfftw3-dev")
endif()

# Проверка наличия библиотек
if(NOT SoapySDR_FOUND)
    message(FATAL_ERROR "SoapySDR not found!")
endif()

if(NOT SDL2_FOUND)
    message(FATAL_ERROR "SDL2 not found!")
endif()

# Проверка наличия файлов ImGui/ImPlot
set(IMGUI_REQUIRED_FILES
    ${THIRD_PARTY_DIR}/imgui/imgui.cpp
    ${THIRD_PARTY_DIR}/imgui/imgui.h
    ${THIRD_PARTY_DIR}/imgui/backends/imgui_impl_sdl2.cpp
    ${THIRD_PARTY_DIR}/imgui/backends/imgui_impl_opengl3.cpp
    ${THIRD_PARTY_DIR}/implot/implot.cpp
    ${THIRD_PARTY_DIR}/implot/implot.h
)

foreach(FILE ${IMGUI_REQUIRED_FILES})
    if(NOT EXISTS ${FILE})
        message(WARNING "Required file not found: ${FILE}")
    endif()
endforeach()

# ============================================================================
# ImGui библиотека
# ============================================================================
if(BUILD_RX OR BUILD_MAIN)
    add_library(imgui STATIC
        ${THIRD_PARTY_DIR}/imgui/imgui.cpp
        ${THIRD_PARTY_DIR}/imgui/imgui_draw.cpp
        ${THIRD_PARTY_DIR}/imgui/imgui_tables.cpp
        ${THIRD_PARTY_DIR}/imgui/imgui_widgets.cpp
        ${THIRD_PARTY_DIR}/imgui/backends/imgui_impl_sdl2.cpp
        ${THIRD_PARTY_DIR}/imgui/backends/imgui_impl_opengl3.cpp
    )
    
    target_include_directories(imgui PUBLIC
        ${THIRD_PARTY_DIR}/imgui
        ${THIRD_PARTY_DIR}/imgui/backends
        ${SDL2_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIRS}
    )
    
    target_compile_definitions(imgui PRIVATE
        IMGUI_IMPL_OPENGL_LOADER_GLEW
    )
    
    set_target_properties(imgui PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
    
    message(STATUS "ImGui library configured")
endif()

# ============================================================================
# ImPlot библиотека
# ============================================================================
if(BUILD_RX OR BUILD_MAIN)
    add_library(implot STATIC
        ${THIRD_PARTY_DIR}/implot/implot.cpp
        ${THIRD_PARTY_DIR}/implot/implot_items.cpp
    )
    
    target_include_directories(implot PUBLIC
        ${THIRD_PARTY_DIR}/implot
    )
    
    target_link_libraries(implot PRIVATE imgui)
    
    set_target_properties(implot PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
    
    message(STATUS "ImPlot library configured")
endif()

# ============================================================================
# Общие настройки для всех исполняемых файлов
# ============================================================================
function(setup_common_target TARGET)
    target_include_directories(${TARGET} PRIVATE
        ${THIRD_PARTY_DIR}/imgui
        ${THIRD_PARTY_DIR}/implot
        ${SDL2_INCLUDE_DIRS}
        ${GLEW_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIRS}
        ${SoapySDR_INCLUDE_DIRS}
        ${FFTW3_INCLUDE_DIRS}
    )
    
    target_compile_options(${TARGET} PRIVATE
        -Wall
        -Wextra
        -Wno-unused-parameter
        -Wno-missing-field-initializers
    )
endfunction()

# ============================================================================
# Исполняемый файл: Передатчик (bpsk_tx)
# ============================================================================
if(BUILD_TX)
    if(EXISTS ${DEV_DIR}/bpsk_tx.cpp)
        add_executable(bpsk_tx ${DEV_DIR}/bpsk_tx.cpp)
        setup_common_target(bpsk_tx)
        
        target_link_libraries(bpsk_tx PRIVATE
            ${SoapySDR_LIBRARIES}
            Threads::Threads
            m
        )
        
        message(STATUS "TX target configured: bpsk_tx")
    else()
        message(WARNING "bpsk_tx.cpp not found at ${DEV_DIR}/bpsk_tx.cpp")
    endif()
endif()

# ============================================================================
# Исполняемый файл: Приемник с GUI (bpsk_rx)
# ============================================================================
if(BUILD_RX)
    if(EXISTS ${DEV_DIR}/bpsk_rx.cpp)
        add_executable(bpsk_rx ${DEV_DIR}/bpsk_rx.cpp)
        setup_common_target(bpsk_rx)
        
        target_link_libraries(bpsk_rx PRIVATE
            imgui
            implot
            ${SDL2_LIBRARIES}
            ${OPENGL_LIBRARIES}
            ${GLEW_LIBRARIES}
            ${SoapySDR_LIBRARIES}
            ${FFTW3_LIBRARIES}
            Threads::Threads
            m
            GL
        )
        
        # Для корректной линковки GLEW на некоторых системах
        if(UNIX AND NOT APPLE)
            target_link_libraries(bpsk_rx PRIVATE GLU)
        endif()
        
        message(STATUS "RX target configured: bpsk_rx")
    else()
        message(WARNING "bpsk_rx.cpp not found at ${DEV_DIR}/bpsk_rx.cpp")
    endif()
endif()

# ============================================================================
# Исполняемый файл: Оригинальная версия (main)
# ============================================================================
if(BUILD_MAIN)
    if(EXISTS ${DEV_DIR}/main.cpp)
        add_executable(main ${DEV_DIR}/main.cpp)
        setup_common_target(main)
        
        target_link_libraries(main PRIVATE
            imgui
            implot
            ${SDL2_LIBRARIES}
            ${OPENGL_LIBRARIES}
            ${GLEW_LIBRARIES}
            ${SoapySDR_LIBRARIES}
            ${FFTW3_LIBRARIES}
            Threads::Threads
            m
            GL
        )
        
        message(STATUS "Original main target configured")
    else()
        message(WARNING "main.cpp not found at ${DEV_DIR}/main.cpp")
    endif()
endif()

# ============================================================================
# Дополнительные цели
# ============================================================================

# Цель для отображения информации
add_custom_target(info
    COMMAND ${CMAKE_COMMAND} -E echo "========================================"
    COMMAND ${CMAKE_COMMAND} -E echo "PlutoSDR Project Information"
    COMMAND ${CMAKE_COMMAND} -E echo "========================================"
    COMMAND ${CMAKE_COMMAND} -E echo "Build directory: ${CMAKE_BINARY_DIR}"
    COMMAND ${CMAKE_COMMAND} -E echo "SoapySDR: ${SoapySDR_LIBRARIES}"
    COMMAND ${CMAKE_COMMAND} -E echo "SDL2: ${SDL2_LIBRARIES}"
    COMMAND ${CMAKE_COMMAND} -E echo "FFTW3: ${FFTW3_LIBRARIES}"
    COMMAND ${CMAKE_COMMAND} -E echo "========================================"
    COMMAND ${CMAKE_COMMAND} -E echo "Available targets:"
    COMMAND ${CMAKE_COMMAND} -E echo "  make bpsk_tx  - Build transmitter"
    COMMAND ${CMAKE_COMMAND} -E echo "  make bpsk_rx  - Build receiver with GUI"
    COMMAND ${CMAKE_COMMAND} -E echo "  make main     - Build original version"
    COMMAND ${CMAKE_COMMAND} -E echo "  make info     - Show this info"
    COMMAND ${CMAKE_COMMAND} -E echo "========================================"
    COMMENT "Display project information"
)

# Инструкция по сборке
message(STATUS "")
message(STATUS "========================================")
message(STATUS "Configuration complete")
message(STATUS "To build:")
message(STATUS "  make bpsk_tx  - Build transmitter")
message(STATUS "  make bpsk_rx  - Build receiver with GUI")
message(STATUS "  make main     - Build original version")
message(STATUS "  make all      - Build all targets")
message(STATUS "  make info     - Show project info")
message(STATUS "========================================")