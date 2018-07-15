cd /d %~dp0
:LOOP
del /q .\kifu
python randkifu_policymove.py
python make_kifu_list.py ./kifu kifulist
python policymove_learn.py
goto :LOOP