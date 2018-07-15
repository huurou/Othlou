cd /d %~dp0
:LOOP
del /q .\kifu
python randkifu_value.py
python make_kifu_list.py ./kifu kifulist
python value_conv_learn.py
goto :LOOP