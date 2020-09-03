outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

project "NN-Toolkit"
    kind "StaticLib"
    location "../build"

    targetname ("nn-toolkit")
    targetdir ("bin/" .. outputdir)
    objdir ("obj/" .. outputdir)
    
    includedirs { "include" }
    
    files {
        "src/**.cpp",
        "src/**.hpp",
        "include/**.hpp"
    }
