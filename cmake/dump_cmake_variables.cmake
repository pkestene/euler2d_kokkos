# adapted from
# https://stackoverflow.com/questions/9298278/cmake-print-out-all-accessible-variables-in-a-script

# dump_cmake_variable: dump all variables formatted as name=value
function(dump_cmake_variables)
  get_cmake_property(_variableNames VARIABLES)
  list(SORT _variableNames)
  foreach(variableName ${_variableNames})
    if(ARGV0)
      unset(MATCHED)
      string(REGEX MATCH ${ARGV0} MATCHED ${variableName})
      if(NOT MATCHED)
        continue()
      endif()
    endif()
    message(STATUS "${variableName}=${${variableName}}")
  endforeach()
endfunction()
