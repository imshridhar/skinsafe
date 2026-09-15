import pytest
from backend.app.utils.security import validate_password_strength, validate_image_magic_bytes


def test_password_validation():
    """Verify password entropy validation rules."""
    # Valid password
    valid, err = validate_password_strength("SkinSafe#2026")
    assert valid is True
    assert err is None
    
    # Too short (<8)
    valid, err = validate_password_strength("Short#1")
    assert valid is False
    
    # Missing uppercase
    valid, err = validate_password_strength("skinsafe#2026")
    assert valid is False
    
    # Missing digit
    valid, err = validate_password_strength("SkinSafe#Pass")
    assert valid is False
    
    # Missing special character
    valid, err = validate_password_strength("SkinSafe2026")
    assert valid is False
    
    # Contains whitespace
    valid, err = validate_password_strength("Skin Safe#2026")
    assert valid is False


def test_magic_bytes_validation():
    """Verify file signature detection."""
    # Valid JPEG header
    jpeg_header = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    valid, mime = validate_image_magic_bytes(jpeg_header)
    assert valid is True
    assert mime == "image/jpeg"
    
    # Valid PNG header
    png_header = b"\x89PNG\r\n\x1a\n\x00\x00"
    valid, mime = validate_image_magic_bytes(png_header)
    assert valid is True
    assert mime == "image/png"
    
    # Invalid / Executable header (MZ header for PE/exe)
    exe_header = b"MZ\x90\x00\x03\x00\x00\x00"
    valid, err = validate_image_magic_bytes(exe_header)
    assert valid is False
