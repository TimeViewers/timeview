@ECHO OFF

SET DIR=%~dp0
SET DIR=%DIR:~0,-1%
CALL :RESOLVE "%DIR%\..\..\.." ROOT

if exist "%ROOT%\miniconda\python" (
        ECHO Already installed
        EXIT /B
) else (
        SETLOCAL

        if Not Exist "%ROOT%\miniconda" (
            echo Downloading anaconda data science platform
            "%DIR%\curl.exe" -o "%DIR%\miniconda.exe" https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe -k
            echo Installing anaconda
            start /wait "" "%DIR%\miniconda.exe" /InstallationType=JustMe /RegisterPython=0 /S /AddToPath=0 /D=%ROOT%\miniconda
            del "%DIR%\miniconda.exe"
        )
        if Exist "%ROOT%\miniconda" (
        echo Creating environment, please wait
            "%ROOT%\miniconda\Scripts\conda-env" create --force -p "%ROOT%\miniconda\envs\timeview" -f "%ROOT%\environment.yml"
))

ENDLOCAL
GOTO :EOF

:RESOLVE
SET %2=%~f1
GOTO :EOF
