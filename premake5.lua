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

project "Matrix-test"
    kind "ConsoleApp"
    targetdir ("tests/bin/" .. outputdir .. "/%{prj.name}")
    objdir ("tests/obj/" .. outputdir .. "/%{prj.name}")
    files { "tests/Matrix.cpp" }
    links { "NN-Toolkit" } 

project "Perceptron-test"
    kind "ConsoleApp"
    targetdir ("tests/bin/" .. outputdir .. "/%{prj.name}")
    objdir ("tests/obj/" .. outputdir .. "/%{prj.name}")
    files { "tests/Perceptron.cpp" }
    links { "NN-Toolkit" } 

project "MultilayerPerceptron-test"
    kind "ConsoleApp"
    targetdir ("tests/bin/" .. outputdir .. "/%{prj.name}")
    objdir ("tests/obj/" .. outputdir .. "/%{prj.name}")
    files { "tests/MultilayerPerceptron.cpp" }
    links { "NN-Toolkit" } 

project "Conv2D-test"
    kind "ConsoleApp"
    targetdir ("tests/bin/" .. outputdir .. "/%{prj.name}")
    objdir ("tests/obj/" .. outputdir .. "/%{prj.name}")
    files { "tests/Conv2D.cpp" }
    links { "NN-Toolkit" } 