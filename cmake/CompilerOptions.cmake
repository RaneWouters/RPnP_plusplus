function(set_global_target_properties target)
  target_compile_features(${target} PUBLIC cxx_std_17)
  target_compile_definitions(${target} PUBLIC $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:_USE_MATH_DEFINES>)
  target_compile_options(
    ${target}
    PRIVATE
            $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4>
            $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/WX>
            $<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-fcolor-diagnostics>
            $<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-Wall>
            $<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-Wextra>
            $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-fdiagnostics-color=always>
            $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wall>
            $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wextra>
            $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-pedantic>)
endfunction()
