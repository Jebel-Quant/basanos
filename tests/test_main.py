from basanos.main import say_hello, main
from io import StringIO
import sys


def test_say_hello():
    assert say_hello("World") == "Hello, World!"


def test_say_hello_custom_name():
    assert say_hello("Alice") == "Hello, Alice!"


def test_main_prints_hello_world(capsys):
    main()
    captured = capsys.readouterr()
    assert captured.out == "Hello, World!\n"
