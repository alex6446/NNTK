@echo off
pushd %~dp0\..\
call vendor\premake\bin\windows\premake5.exe vs2019
popd
PAUSE
