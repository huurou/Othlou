cd /d %~dp0
: loop
del /q .\kifu
python randkifu_legal.py
python make_kifu_list.py ./kifu kifulist
python legal_learn.py
goto :loop