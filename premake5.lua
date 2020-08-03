workspace "NN-Toolkit"
	architecture "x86_64"
    configurations { "Debug", "Release" }
    
    includedirs { "include" }
    
    filter "configurations:Debug"
        defines { "DEBUG" }
        symbols "On"

    filter "configurations:Release"
        defines { "NDEBUG" }
        optimize "On"

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

project "NN-Toolkit"
    kind "StaticLib"

    targetname ("nn-toolkit")
    targetdir ("bin/" .. outputdir)
    objdir ("obj/" .. outputdir)
    
    files {
        "src/**.cpp",
        "src/**.hpp",
        "include/**.hpp"
    }

    


project "Functions-test"
    kind "ConsoleApp"
    targetdir ("tests/bin/" .. outputdir .. "/%{prj.name}")
    objdir ("tests/obj/" .. outputdir .. "/%{prj.name}")
    files { "tests/Functions.cpp" }
    links { "NN-Toolkit" } 

project "ErrorHandling-test"
    kind "ConsoleApp"
    targetdir ("tests/bin/" .. outputdir .. "/%{prj.name}")
    objdir ("tests/obj/" .. outputdir .. "/%{prj.name}")
    files { "tests/ErrorHandling.cpp" }
    links { "NN-Toolkit" } 
