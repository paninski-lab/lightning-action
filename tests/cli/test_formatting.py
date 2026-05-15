"""Tests for CLI formatting utilities."""

import argparse

import pytest

from lightning_action.cli.formatting import ArgumentParser, HelpFormatter, SubArgumentParser


class TestArgumentParserPrintHelp:
    """Test the print_help method of ArgumentParser."""

    def test_print_help_includes_welcome_text(self, capsys):
        """Test that main parser prints welcome banner."""
        parser = ArgumentParser(prog='test')
        parser.print_help()
        captured = capsys.readouterr()
        assert 'lightning-action' in captured.out

    def test_sub_parser_omits_welcome_text(self, capsys):
        """Test that sub-parsers skip the welcome banner."""
        parser = SubArgumentParser(prog='test')
        parser.print_help()
        captured = capsys.readouterr()
        assert 'lightning-action' not in captured.out


class TestArgumentParserError:
    """Test the error method of ArgumentParser."""

    def test_error_exits_with_code_2(self):
        """Test that error raises SystemExit with code 2."""
        parser = ArgumentParser(prog='test')
        with pytest.raises(SystemExit) as exc_info:
            parser.error('something went wrong')
        assert exc_info.value.code == 2

    def test_error_writes_message_to_stderr(self, capsys):
        """Test that error message appears on stderr."""
        parser = ArgumentParser(prog='test')
        with pytest.raises(SystemExit):
            parser.error('something went wrong')
        assert 'something went wrong' in capsys.readouterr().err


class TestHelpFormatterSplitLines:
    """Test the _split_lines method of HelpFormatter."""

    def test_preserves_newlines(self):
        """Test that embedded newlines create separate lines."""
        formatter = HelpFormatter(prog='test')
        lines = formatter._split_lines('line one\nline two', 80)
        assert lines == ['line one', 'line two']

    def test_empty_line_becomes_empty_string(self):
        """Test that an empty paragraph produces an empty string entry."""
        formatter = HelpFormatter(prog='test')
        lines = formatter._split_lines('before\n\nafter', 80)
        assert '' in lines

    def test_wraps_long_text(self):
        """Test that text longer than width is wrapped."""
        formatter = HelpFormatter(prog='test')
        long_text = 'word ' * 30
        lines = formatter._split_lines(long_text, 40)
        assert len(lines) > 1
        for line in lines:
            assert len(line) <= 40


class TestHelpFormatterFillText:
    """Test the _fill_text method of HelpFormatter."""

    def test_applies_indent_to_each_line(self):
        """Test that indent is prepended to every line."""
        formatter = HelpFormatter(prog='test')
        result = formatter._fill_text('hello\nworld', 80, '  ')
        for line in result.splitlines():
            assert line.startswith('  ')

    def test_multiline_output(self):
        """Test that multiline input produces multiline output."""
        formatter = HelpFormatter(prog='test')
        result = formatter._fill_text('line one\nline two', 80, '')
        assert '\n' in result


class TestHelpFormatterFormatAction:
    """Test the _format_action method of HelpFormatter."""

    def test_adds_trailing_newline(self):
        """Test that an extra newline is appended after each action."""
        parser = argparse.ArgumentParser(formatter_class=HelpFormatter, prog='test')
        parser.add_argument('--foo', help='foo help')
        formatter = parser._get_formatter()
        action = next(a for a in parser._actions if '--foo' in a.option_strings)
        result = formatter._format_action(action)
        assert result.endswith('\n\n')
