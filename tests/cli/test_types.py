"""Tests for CLI type validators."""

import tempfile
from pathlib import Path

import pytest

from lightning_action.cli.types import config_file, output_dir, valid_dir, valid_file


class TestValidFile:
    """Test valid_file validator."""

    def test_valid_file_existing_file(self):
        """Test valid_file with existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = Path(f.name)
        
        try:
            result = valid_file(temp_file)
            assert result == temp_file
            assert result.is_file()
        finally:
            temp_file.unlink()

    def test_valid_file_string_path(self):
        """Test valid_file with string path."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file_str = f.name
            temp_file = Path(temp_file_str)
        
        try:
            result = valid_file(temp_file_str)
            assert result == temp_file
            assert result.is_file()
        finally:
            temp_file.unlink()

    def test_valid_file_nonexistent(self):
        """Test valid_file with nonexistent file."""
        with pytest.raises(IOError, match='File does not exist'):
            valid_file('/nonexistent/file.txt')

    def test_valid_file_directory(self):
        """Test valid_file with directory instead of file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(IOError, match='Not a file'):
                valid_file(temp_dir)


class TestValidDir:
    """Test valid_dir validator."""

    def test_valid_dir_existing_directory(self):
        """Test valid_dir with existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            result = valid_dir(temp_path)
            assert result == temp_path
            assert result.is_dir()

    def test_valid_dir_string_path(self):
        """Test valid_dir with string path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = valid_dir(temp_dir)
            assert result == Path(temp_dir)
            assert result.is_dir()

    def test_valid_dir_nonexistent(self):
        """Test valid_dir with nonexistent directory."""
        with pytest.raises(IOError, match='Directory does not exist'):
            valid_dir('/nonexistent/directory')

    def test_valid_dir_file(self):
        """Test valid_dir with file instead of directory."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = Path(f.name)
        
        try:
            with pytest.raises(IOError, match='Not a directory'):
                valid_dir(temp_file)
        finally:
            temp_file.unlink()


class TestConfigFile:
    """Test config_file validator."""

    def test_config_file_yaml_extension(self):
        """Test config_file with .yaml extension."""
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            temp_file = Path(f.name)
        
        try:
            result = config_file(temp_file)
            assert result == temp_file
        finally:
            temp_file.unlink()

    def test_config_file_yml_extension(self):
        """Test config_file with .yml extension."""
        with tempfile.NamedTemporaryFile(suffix='.yml', delete=False) as f:
            temp_file = Path(f.name)
        
        try:
            result = config_file(temp_file)
            assert result == temp_file
        finally:
            temp_file.unlink()

    def test_config_file_invalid_extension(self):
        """Test config_file with invalid extension."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_file = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match='Config file must be YAML'):
                config_file(temp_file)
        finally:
            temp_file.unlink()

    def test_config_file_nonexistent(self):
        """Test config_file with nonexistent file."""
        with pytest.raises(IOError, match='File does not exist'):
            config_file('/nonexistent/config.yaml')

    def test_config_file_string_path(self):
        """Test config_file with string path."""
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            temp_file_str = f.name
            temp_file = Path(temp_file_str)
        
        try:
            result = config_file(temp_file_str)
            assert result == temp_file
        finally:
            temp_file.unlink()


class TestOutputDir:
    """Test output_dir validator."""

    def test_output_dir_existing_directory(self):
        """Test output_dir with existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            result = output_dir(temp_path)
            assert result == temp_path
            assert result.is_dir()

    def test_output_dir_creates_new_directory(self):
        """Test output_dir creates new directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / 'new_subdir'
            assert not new_dir.exists()
            
            result = output_dir(new_dir)
            assert result == new_dir
            assert result.is_dir()

    def test_output_dir_creates_nested_directories(self):
        """Test output_dir creates nested directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = Path(temp_dir) / 'level1' / 'level2' / 'level3'
            assert not nested_dir.exists()
            
            result = output_dir(nested_dir)
            assert result == nested_dir
            assert result.is_dir()

    def test_output_dir_string_path(self):
        """Test output_dir with string path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir_str = str(Path(temp_dir) / 'string_subdir')
            new_dir = Path(new_dir_str)
            assert not new_dir.exists()
            
            result = output_dir(new_dir_str)
            assert result == new_dir
            assert result.is_dir()

    def test_output_dir_existing_file_error(self):
        """Test output_dir behavior when path exists as file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = Path(f.name)
        
        try:
            # This should raise an error because the path exists as a file
            with pytest.raises(FileExistsError):
                output_dir(temp_file)
        except OSError:
            # Some systems might raise different OS errors
            pass
        finally:
            temp_file.unlink()
