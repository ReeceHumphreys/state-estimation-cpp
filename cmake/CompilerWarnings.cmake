function(set_project_warnings target)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
        target_compile_options(${target} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wsign-conversion
        )
    elseif (MSVC)
        target_compile_options(${target} PRIVATE
            /W4
            /permissive-
        )
    endif()
endfunction()
