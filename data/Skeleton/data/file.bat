@echo off

if exist list.txt del list.txt /q
:input
cls
set input=:
set /p input=Please input path:
set "input=%input:"=%"
:: �������Ϊ�ж�%input%���Ƿ�������ţ������޳���
if "%input%"==":" goto input
if not exist "%input%" goto input
for %%i in ("%input%") do if /i "%%~di"==%%i goto input
pushd %cd%
cd /d "%input%">nul 2>nul || exit
set cur_dir=%cd%
popd
:: %%~nxiֻ��ʾ�ļ���,%%i��ʾ��·�����ļ���Ϣ
for /f "delims=" %%i in ('dir /b /a-d /s "%input%"') do echo %%i>>list.txt
if not exist list.txt goto no_file
start list.txt
exit

:no_file
cls
echo %cur_dir% Folder does not have a separate document
pause