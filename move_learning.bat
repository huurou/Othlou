cd /d %~dp0
: loop
del /q .\kifu
python kifu_ucb.py
python make_kifu_list.py ./kifu kifulist
python move_learn.py
goto :loop