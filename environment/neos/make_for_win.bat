rem variables

rem GRPC-TOOLS required. Install with `nuget install Grpc.Tools`.
rem Then un-comment and replace [DIRECTORY] with location of files.
rem For example, on Windows, you might have something like:
set COMPILER=Grpc.Tools.1.14.1\tools\windows_x64
rem set COMPILER=[DIRECTORY]

set SRC_DIR=protos
set DST_DIR_N=TeachableNeos\TeachableNeos\Grpc
set DST_DIR_U=UnityMiddleWare2\Assets\VoidEnvironment\Grpc
set PROTO_PATH=protos

rem clean
rd /s /q %DST_DIR_N%
mkdir %DST_DIR_N%
rd /s /q %DST_DIR_U%
mkdir %DST_DIR_U%

rem generate proto objects in python and C#

for %%i in (%SRC_DIR%\*.proto) do (
    %COMPILER%\protoc --proto_path=%PROTO_PATH% --csharp_out=%DST_DIR_N% %%i
    %COMPILER%\protoc --proto_path=%PROTO_PATH% --csharp_out=%DST_DIR_U% %%i
    rem %COMPILER%\protoc --proto_path=%PROTO_PATH% --python_out=%DST_DIR_P% %%i
)

rem grpc

set GRPC=basic_comm.proto

%COMPILER%\protoc --proto_path=%PROTO_PATH% --csharp_out %DST_DIR_N% --grpc_out=%DST_DIR_N% %SRC_DIR%\%GRPC% --plugin=protoc-gen-grpc=%COMPILER%\grpc_csharp_plugin.exe
%COMPILER%\protoc --proto_path=%PROTO_PATH% --csharp_out %DST_DIR_U% --grpc_out=%DST_DIR_U% %SRC_DIR%\%GRPC% --plugin=protoc-gen-grpc=%COMPILER%\grpc_csharp_plugin.exe
rem python -m grpc_tools.protoc --proto_path=%PROTO_PATH% --python_out=%DST_DIR_P% --grpc_python_out=%DST_DIR_P% %SRC_DIR%\%GRPC%

rem Generate the init file for the python module
rem rm -f $DST_DIR_P/$PYTHON_PACKAGE/__init__.py
rem setlocal enabledelayedexpansion
rem for %%i in (%DST_DIR_P%\%PYTHON_PACKAGE%\*.py) do (
rem set FILE=%%~ni
rem echo from .$(basename $FILE) import * >> $DST_DIR_P/$PYTHON_PACKAGE/__init__.py
rem echo from .!FILE! import * >> %DST_DIR_P%\%PYTHON_PACKAGE%\__init__.py
rem )

