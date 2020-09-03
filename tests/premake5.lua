outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

project "Functions-test"
    kind "ConsoleApp"
    location "../build"
    targetdir ("bin/" .. outputdir .. "/%{prj.name}")
    objdir ("obj/" .. outputdir .. "/%{prj.name}")
    includedirs { "../NN-Toolkit/include" }
    files { "Functions.cpp" }
    links { "NN-Toolkit" } 

project "ErrorHandling-test"
    kind "ConsoleApp"
    location "../build"
    targetdir ("bin/" .. outputdir .. "/%{prj.name}")
    objdir ("obj/" .. outputdir .. "/%{prj.name}")
    includedirs { "../NN-Toolkit/include" }
    files { "ErrorHandling.cpp" }
    links { "NN-Toolkit" } 

project "Matrix-test"
    kind "ConsoleApp"
    location "../build"
    targetdir ("bin/" .. outputdir .. "/%{prj.name}")
    objdir ("obj/" .. outputdir .. "/%{prj.name}")
    includedirs { "../NN-Toolkit/include" }
    files { "Matrix.cpp" }
    links { "NN-Toolkit" } 

project "Perceptron-test"
    kind "ConsoleApp"
    location "../build"
    targetdir ("bin/" .. outputdir .. "/%{prj.name}")
    objdir ("obj/" .. outputdir .. "/%{prj.name}")
    includedirs { "../NN-Toolkit/include" }
    files { "Perceptron.cpp" }
    links { "NN-Toolkit" } 

project "MultilayerPerceptron-test"
    kind "ConsoleApp"
    location "../build"
    targetdir ("bin/" .. outputdir .. "/%{prj.name}")
    objdir ("obj/" .. outputdir .. "/%{prj.name}")
    includedirs { "../NN-Toolkit/include" }
    files { "MultilayerPerceptron.cpp" }
    links { "NN-Toolkit" } 

project "Conv2D-test"
    kind "ConsoleApp"
    location "../build"
    targetdir ("bin/" .. outputdir .. "/%{prj.name}")
    objdir ("obj/" .. outputdir .. "/%{prj.name}")
    includedirs { "../NN-Toolkit/include" }
    files { "Conv2D.cpp" }
    links { "NN-Toolkit" } 

project "Flatten-test"
    kind "ConsoleApp"
    location "../build"
    targetdir ("bin/" .. outputdir .. "/%{prj.name}")
    objdir ("obj/" .. outputdir .. "/%{prj.name}")
    includedirs { "../NN-Toolkit/include" }
    files { "Flatten.cpp" }
    links { "NN-Toolkit" } 

project "MaxPooling2D-test"
    kind "ConsoleApp"
    location "../build"
    targetdir ("bin/" .. outputdir .. "/%{prj.name}")
    objdir ("obj/" .. outputdir .. "/%{prj.name}")
    includedirs { "../NN-Toolkit/include" }
    files { "MaxPooling2D.cpp" }
    links { "NN-Toolkit" }

project "AveragePooling2D-test"
    kind "ConsoleApp"
    location "../build"
    targetdir ("bin/" .. outputdir .. "/%{prj.name}")
    objdir ("obj/" .. outputdir .. "/%{prj.name}")
    includedirs { "../NN-Toolkit/include" }
    files { "AveragePooling2D.cpp" }
    links { "NN-Toolkit" }

project "CNN-test"
    kind "ConsoleApp"
    location "../build"
    targetdir ("bin/" .. outputdir .. "/%{prj.name}")
    objdir ("obj/" .. outputdir .. "/%{prj.name}")
    includedirs { "../NN-Toolkit/include" }
    files { "CNN.cpp" }
    links { "NN-Toolkit" } 