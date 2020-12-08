workspace "NN-Toolkit"
    architecture "x86_64"
    configurations { "Debug", "Release" }
        
    filter "configurations:Debug"
        defines { "DEBUG" }
        symbols "On"

    filter "configurations:Release"
        defines { "NDEBUG" }
        optimize "On"

include ".."
include "../tests"
