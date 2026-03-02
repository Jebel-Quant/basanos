"""Tests for basanos.main module."""

from basanos.main import main, say_hello


def test_say_hello():
    """Test say_hello with default name."""
    assert say_hello("World") == "Hello, World!"


def test_say_hello_custom_name():
    """Test say_hello with a custom name."""
    assert say_hello("Alice") == "Hello, Alice!"


def test_main_prints_hello_world(capsys):
    """Test that main prints Hello, World! to stdout."""
    main()
    captured = capsys.readouterr()
    assert captured.out == "Hello, World!\n"
