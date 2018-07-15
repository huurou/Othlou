cd /d %~dp0
set cnt=1
:LOOP
@echo cnt=%cnt%
del /q .\kifu
python othero_modelkifu.py
python make_kifu_list.py ./kifu kifulist
python nnlearnkifu.py
set /a cnt=cnt+1
if %cnt% gtr 30 exit /b
goto :LOOP