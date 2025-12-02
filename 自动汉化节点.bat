@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo    中文本地化文件安装工具
echo ========================================
echo.

:: 获取当前脚本所在目录，然后指向ComfyUI-DD-Translation子目录
set "SCRIPT_DIR=%~dp0"
set "SOURCE_DIR=%SCRIPT_DIR%ComfyUI-DD-Translation\"

:: 检查源文件是否存在
echo 正在检查源文件...
set "FILE_COUNT=0"
for /f "delims=" %%f in ('dir /s /b "%SOURCE_DIR%*.json" 2^>nul') do set /a FILE_COUNT+=1

if %FILE_COUNT% equ 0 (
    echo 错误：ComfyUI-DD-Translation 目录（包含子目录）中没有找到任何 .json 文件
    echo 请确保目录包含需要安装的本地化文件
    pause
    exit /b 1
)

echo ✓ 找到 %FILE_COUNT% 个 .json 文件待安装
echo.

:: 检查ComfyUI-DD-Translation扩展是否存在
echo 正在检查 ComfyUI-DD-Translation 扩展...

:: 向上查找custom_nodes目录
set "CURRENT_DIR=%SCRIPT_DIR%"
set "CUSTOM_NODES_DIR="

:find_custom_nodes
if exist "%CURRENT_DIR%custom_nodes" (
    set "CUSTOM_NODES_DIR=%CURRENT_DIR%custom_nodes"
    goto :found_custom_nodes
)

:: 向上一级目录查找
for %%i in ("%CURRENT_DIR%..") do set "CURRENT_DIR=%%~fi\"
if "%CURRENT_DIR%"=="%CURRENT_DIR:~0,3%" (
    echo 错误：无法找到 custom_nodes 目录
    goto :extension_not_found
)
goto :find_custom_nodes

:found_custom_nodes
set "DD_TRANSLATION_DIR=%CUSTOM_NODES_DIR%\ComfyUI-DD-Translation"

if exist "%DD_TRANSLATION_DIR%" (
echo ✓ 找到 ComfyUI-DD-Translation 扩展
    echo   扩展路径: %DD_TRANSLATION_DIR%
    echo.
    goto :install_files
) else (
    echo ! 未找到 ComfyUI-DD-Translation 扩展
    goto :extension_not_found
)

:install_files
echo 开始安装本地化文件...
echo.

:: 创建目标目录（如果不存在）
set "NODES_TARGET_DIR=%DD_TRANSLATION_DIR%\zh-CN\Nodes"

if not exist "%NODES_TARGET_DIR%" (
    echo 创建目录: %NODES_TARGET_DIR%
    mkdir "%NODES_TARGET_DIR%" 2>nul
    if errorlevel 1 (
        echo 错误：无法创建目录 %NODES_TARGET_DIR%
        pause
        exit /b 1
    )
)

:: 复制文件
echo 正在复制文件...

set "COPY_COUNT=0"
set "ERROR_FILES="

for /f "delims=" %%f in ('dir /s /b "%SOURCE_DIR%*.json" 2^>nul') do (
    set "FILE_PATH=%%~dpf"
    set "FILE_NAME=%%~nxf"
    echo   复制 !FILE_PATH! → !FILE_NAME! 到 !NODES_TARGET_DIR!
    copy "%%f" "%NODES_TARGET_DIR%\" >nul 2>&1
    if errorlevel 1 (
        echo ✗ 复制 !FILE_NAME! 失败
        set "ERROR_FILES=!ERROR_FILES! !FILE_PATH!!FILE_NAME!"
    ) else (
        echo ✓ !FILE_NAME! 复制成功
        set /a COPY_COUNT+=1
    )
)

echo.
if not "!ERROR_FILES!"=="" (
    echo ========================================
    echo      部分文件安装失败！
    echo ========================================
    echo.
    set "DETAIL_TEXT=安装详情："
    echo !DETAIL_TEXT!
    echo   • 成功安装 !COPY_COUNT! 个文件
    echo   • 失败文件：%ERROR_FILES%
    pause
    exit /b 1
)

echo ========================================
echo     本地化文件安装完成！
echo ========================================
echo.
set "DETAIL_TEXT=安装详情："
echo !DETAIL_TEXT!
echo   • 成功安装 !COPY_COUNT! 个文件到 %NODES_TARGET_DIR%
echo.
echo 注意：请重启 ComfyUI 以使本地化生效
echo.
pause
exit /b 0

:extension_not_found
echo.
echo ========================================
echo     未找到 ComfyUI-DD-Translation 扩展
echo ========================================
echo.
echo 要使用节点中文本地化功能，您需要先安装 ComfyUI-DD-Translation 扩展。
echo.
echo 安装方法：
echo   1. 打开 ComfyUI Manager
echo   2. 搜索并安装 "ComfyUI-DD-Translation"
echo.
echo 或者手动安装：
echo   1. 进入 ComfyUI 的 custom_nodes 目录
echo   2. 执行命令：git clone https://github.com/Dontdrunk/ComfyUI-DD-Translation
echo   3. 重启 ComfyUI
echo   4. 再次运行此安装脚本
echo.
echo 扩展地址：
echo   https://github.com/Dontdrunk/ComfyUI-DD-Translation
echo.
echo 安装完成后，请重新运行此脚本来安装本地化文件。
echo.
pause
exit /b 1
