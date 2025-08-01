# SPDX-FileCopyrightText: 2025 DSBL-Dev contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Security gate for detecting bad code patterns.
Part of defense.
"""

import re
import hashlib
from config.settings import Settings

class SecurityGate:
    """Gate blocking dangerous code execution patterns."""
    
    def __init__(self):
        from config.settings import load_settings
        settings = load_settings()
        self.dangerous_patterns = getattr(settings.security, 'dangerous_patterns', [
            "rm -rf", "system(", "eval(", "exec(", "__import__",
            "subprocess", "os.system", "shell=True",
            "password", "token", "secret", "api_key"
        ])
        
        # Whitespace-agnostic regex patterns for evasion detection
        self.evasion_patterns = [
            r'r\s*m\s*-?\s*r\s*f',  # rm -rf with any spacing
            r'system\s*\(',            # system( with spacing
            r'eval\s*\(',              # eval( with spacing  
            r'exec\s*\(',              # exec( with spacing
            r'__import__',               # __import__ (no spacing issues)
            r'subprocess',               # subprocess (no spacing issues)
            r'os\.system',              # os.system (literal dot)
            r'shell\s*=\s*True',       # shell=True with spacing
        ]
    
    def process(self, content: str, context: str, audit_logger) -> bool:
        """
        Process content through security gate.
        Returns True if content is safe, False if blocked.
        """
        content_lower = content.lower()
        
        # Check whitespace-agnostic evasion patterns
        normalized_content = re.sub(r'\s+', '', content_lower)  # Remove all whitespace
        
        for pattern in self.evasion_patterns:
            if re.search(pattern, content_lower, re.IGNORECASE):
                audit_logger.log_event("GATE_DECISION", {
                    "gate": "sec_clean",
                    "result": "BLOCKED",
                    "reason": f"Contains evasion pattern: {pattern}",
                    "content_hash": hashlib.sha256(content.encode()).hexdigest()[:10],
                    "detection_method": "regex_evasion"
                })
                return False
        
        # Also check normalized content for simple patterns
        for pattern in self.dangerous_patterns:
            normalized_pattern = re.sub(r'\s+', '', pattern.lower())
            if normalized_pattern in normalized_content:
                audit_logger.log_event("GATE_DECISION", {
                    "gate": "sec_clean",
                    "result": "BLOCKED",
                    "reason": f"Contains spaced evasion of: {pattern}",
                    "content_hash": hashlib.sha256(content.encode()).hexdigest()[:10],
                    "detection_method": "normalized_string"
                })
                return False
        
        audit_logger.log_event("GATE_DECISION", {
            "gate": "sec_clean", 
            "result": "ALLOWED",
            "reason": "No dangerous patterns detected",
            "content_hash": hashlib.sha256(content.encode()).hexdigest()[:10]
        })
        return True