@ECHO OFF

SET DIR=%~dp0
CALL :RESOLVE "%DIR%\..\..\.." ROOT

CALL "%ROOT%\miniconda\Scripts\activate" timeview

"%ROOT%\miniconda\envs\timeview\Library\bin\git" pull

CALL "%ROOT\bin\windows\ext\install.bat"


GOTO :EOF

:RESOLVE
SET %2=%~f1
GOTO :EOF
