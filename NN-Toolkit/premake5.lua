project "NN-Toolkit-static-lib"
    kind "StaticLib"
    location "../build"

    targetname ("nn-toolkit")
    targetdir ("bin/")
    objdir ("obj/")
    
    includedirs { "include" }
    
    files {
        "src/**.cpp",
        "src/**.hpp",
        "include/**.hpp"
    }
    
    filter "configurations:Debug"
        targetsuffix ("-debug")
        
project "NN-Toolkit-shared-lib"
    kind "SharedLib"
    location "../build"

    targetname ("nn-toolkit")
    targetdir ("bin/")
    objdir ("obj/")
    
    includedirs { "include" }
    
    files {
        "src/**.cpp",
        "src/**.hpp",
        "include/**.hpp"
    }
    
    filter "configurations:Debug"
        targetsuffix ("-debug")
   