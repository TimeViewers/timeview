@ECHO OFF

SET DIR=%~dp0
CALL :RESOLVE "%DIR%\..\..\.." ROOT
msg /time:3 "%username%" Starting TimeView, please wait...

CALL "%ROOT%\miniconda\Scripts\activate" timeview

CD "%ROOT%"
"%ROOT%\miniconda\envs\timeview\python" -O -m timeview
CD %DIR%

GOTO :EOF

:RESOLVE
SET %2=%~f1
GOTO :EOF

