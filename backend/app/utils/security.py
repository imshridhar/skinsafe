import re
from typing import Tuple, Optional


def validate_image_magic_bytes(header: bytes) -> Tuple[bool, Optional[str]]:
    """Validates image magic bytes to prevent malicious file uploads disguised as JPEGs/PNGs."""
    if len(header) < 8:
        return False, "File too small or header truncated"
        
    # JPEG magic bytes: FF D8 FF
    if header.startswith(b"\xff\xd8\xff"):
        return True, "image/jpeg"
        
    # PNG magic bytes: 89 50 4E 47 0D 0A 1A 0A
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return True, "image/png"
        
    return False, "Unsupported file signature (Only JPEG and PNG allowed)"


def validate_password_strength(password: str) -> Tuple[bool, Optional[str]]:
    """Enforces strict password rules in accordance with Agent.md:

    - 8 to 20 characters
    - At least 1 lowercase (a-z)
    - At least 1 uppercase (A-Z)
    - At least 1 digit (0-9)
    - At least 1 special character
    - Printable ASCII only (no whitespace or emoji)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if len(password) > 20:
        return False, "Password must be at most 20 characters long"
    if " " in password:
        return False, "Password must not contain whitespace"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter (a-z)"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter (A-Z)"
    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least one digit (0-9)"
    if not re.search(r"[^A-Za-z0-9]", password):
        return False, "Password must contain at least one special character"
        
    # Printable ASCII verification
    try:
        password.encode("ascii")
    except UnicodeEncodeError:
        return False, "Password must contain printable ASCII characters only (no emoji)"
        
    return True, None
