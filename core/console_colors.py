"""
Console formatting utilities for experiment output.
"""

class Colors:
    """Symbol-based formatting for console output."""
    
    @staticmethod
    def civil(text: str) -> str:
        """Format text for CIVIL gate operations."""
        if "BLOCKED" in text:
            return f"🚫 {text}"
        else:
            return f"✅ {text}"
    
    @staticmethod
    def security(text: str) -> str:
        """Format text for Security gate operations."""
        if "BLOCKED" in text:
            return f"🛡️❌ {text}"
        elif "BYPASS" in text:
            return f"⚠️ {text}"
        else:
            return f"🛡️✅ {text}"
    
    @staticmethod
    def api_call(text: str) -> str:
        """Format text for API call indicators."""
        return f"🌐 {text}"
    
    @staticmethod
    def local(text: str) -> str:
        """Format text for local processing."""
        return f"💻 {text}"
    
    @staticmethod
    def blocked(text: str) -> str:
        """Format text for blocked content."""
        return f"🚫 {text}"
    
    @staticmethod
    def allowed(text: str) -> str:
        """Format text for allowed content."""
        return f"✅ {text}"
    
    @staticmethod
    def system(text: str) -> str:
        """Format text for system messages."""
        return f"⚙️ {text}"
    
    @staticmethod
    def ticket_header(text: str) -> str:
        """Format text for ticket headers."""
        if "TICKET" in text:
            return f"🎫 {text}"
        else:
            return text
    
    @staticmethod
    def debug(text: str) -> str:
        """Format text for debug information."""
        return f"🐛 {text}"
    
    @staticmethod
    def vote(text: str) -> str:
        """Format text for vote calculations."""
        return f"🗳️ {text}"
    
    @staticmethod
    def state(text: str) -> str:
        """Format text for state changes."""
        return f"🔄 {text}"
    
    @staticmethod
    def timing(text: str) -> str:
        """Format text for timing information."""
        return f"⏱️ {text}"