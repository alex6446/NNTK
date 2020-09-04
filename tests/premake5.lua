group "Tests"

    project "Functions-test"
        kind "ConsoleApp"
        location "../build"
        targetdir ("../bin/%{cfg.buildcfg}")
        objdir ("../obj/%{cfg.buildcfg}/%{prj.name}")
        includedirs { "../NN-Toolkit/include" }
        files { "Functions.cpp" }
        links { "NN-Toolkit-static-lib" } 

    project "ErrorHandling-test"
        kind "ConsoleApp"
        location "../build"
        targetdir ("../bin/%{cfg.buildcfg}")
        objdir ("../obj/%{cfg.buildcfg}/%{prj.name}")
        includedirs { "../NN-Toolkit/include" }
        files { "ErrorHandling.cpp" }
        links { "NN-Toolkit-static-lib" } 

    project "Matrix-test"
        kind "ConsoleApp"
        location "../build"
        targetdir ("../bin/%{cfg.buildcfg}")
        objdir ("../obj/%{cfg.buildcfg}/%{prj.name}")
        includedirs { "../NN-Toolkit/include" }
        files { "Matrix.cpp" }
        links { "NN-Toolkit-static-lib" } 

    project "Perceptron-test"
        kind "ConsoleApp"
        location "../build"
        targetdir ("../bin/%{cfg.buildcfg}")
        objdir ("../obj/%{cfg.buildcfg}/%{prj.name}")
        includedirs { "../NN-Toolkit/include" }
        files { "Perceptron.cpp" }
        links { "NN-Toolkit-static-lib" } 

    project "MultilayerPerceptron-test"
        kind "ConsoleApp"
        location "../build"
        targetdir ("../bin/%{cfg.buildcfg}")
        objdir ("../obj/%{cfg.buildcfg}/%{prj.name}")
        includedirs { "../NN-Toolkit/include" }
        files { "MultilayerPerceptron.cpp" }
        links { "NN-Toolkit-static-lib" } 

    project "Conv2D-test"
        kind "ConsoleApp"
        location "../build"
        targetdir ("../bin/%{cfg.buildcfg}")
        objdir ("../obj/%{cfg.buildcfg}/%{prj.name}")
        includedirs { "../NN-Toolkit/include" }
        files { "Conv2D.cpp" }
        links { "NN-Toolkit-static-lib" } 

    project "Flatten-test"
        kind "ConsoleApp"
        location "../build"
        targetdir ("../bin/%{cfg.buildcfg}")
        objdir ("../obj/%{cfg.buildcfg}/%{prj.name}")
        includedirs { "../NN-Toolkit/include" }
        files { "Flatten.cpp" }
        links { "NN-Toolkit-static-lib" } 

    project "MaxPooling2D-test"
        kind "ConsoleApp"
        location "../build"
        targetdir ("../bin/%{cfg.buildcfg}")
        objdir ("../obj/%{cfg.buildcfg}/%{prj.name}")
        includedirs { "../NN-Toolkit/include" }
        files { "MaxPooling2D.cpp" }
        links { "NN-Toolkit-static-lib" }

    project "AveragePooling2D-test"
        kind "ConsoleApp"
        location "../build"
        targetdir ("../bin/%{cfg.buildcfg}")
        objdir ("../obj/%{cfg.buildcfg}/%{prj.name}")
        includedirs { "../NN-Toolkit/include" }
        files { "AveragePooling2D.cpp" }
        links { "NN-Toolkit-static-lib" }

    project "CNN-test"
        kind "ConsoleApp"
        location "../build"
        targetdir ("../bin/%{cfg.buildcfg}")
        objdir ("../obj/%{cfg.buildcfg}/%{prj.name}")
        includedirs { "../NN-Toolkit/include" }
        files { "CNN.cpp" }
        links { "NN-Toolkit-static-lib" } 

group ""