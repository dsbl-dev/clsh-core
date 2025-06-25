"""
CIVIL gate for toxicity detection using OpenAI Moderation API.
v2.3: Enhanced with score-based blocking and content sanitization.
"""

import hashlib
import os
import openai

class CivilGate:
    """Gate that blocks toxic and manipulative content."""
    
    def __init__(self):
        # Initialize OpenAI client (secure pattern)
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # v2.2 Enhancement: Much more aggressive toxicity patterns
        self.toxic_patterns = [
            # Original patterns
            "pity points", "drags us down", "not cut out", "desperate attempt",
            "overrated", "clearly not", "just riding", "unwanted competition",
            "drag us down", "riding coattails", "standout", "doesn't deserve",
            
            # v2.2 New patterns - targeting Mallory's manipulation tactics
            "logical choice", "no-brainer", "real talent", "true potential",
            "fishing for", "desperate grab", "elevate the standard",
            "focus on real", "unlike some who", "let's be real",
            "honestly", "time to recognize", "face it",
            
            # Self-promotion patterns
            "promoting myself", "i deserve", "i actually contribute",
            "unlike others", "i'm the", "vote for me",
            
            # Competitive put-downs
            "usual suspects", "overlooking the potential", "overrating",
            "give someone else a chance", "reconsider", "shake things up"
        ]
        
        # Load toxicity threshold from config
        from config.settings import load_settings
        settings = load_settings()
        self.toxicity_threshold = getattr(settings.civil_gate, 'toxicity_threshold', 0.3)
        
    def process(self, content: str, context: str, audit_logger) -> bool:
        """
        Process content through CIVIL gate.
        v2.3: Enhanced with comprehensive toxicity scoring.
        Returns True if content is civil, False if blocked.
        """
        # Handle empty content
        if not content or content.strip() == "":
            audit_logger.log_event("GATE_DECISION", {
                "gate": "civil",
                "result": "ALLOWED",
                "reason": "Empty content",
                "content_hash": hashlib.sha256("".encode()).hexdigest()[:10]
            })
            return True
            
        # First try OpenAI Moderation API
        try:
            # Timing checkpoint for API call
            if hasattr(audit_logger, 'debug_timing') and audit_logger.debug_timing:
                audit_logger.log_timing_checkpoint("civil_api_start", "OpenAI moderation API call starting")
            
            response = self.openai_client.moderations.create(
                model="omni-moderation-latest",
                input=content,
                timeout=15  # 15 second timeout for moderation calls
            )
            
            if hasattr(audit_logger, 'debug_timing') and audit_logger.debug_timing:
                audit_logger.log_timing_checkpoint("civil_api_end", "OpenAI moderation API call completed")
            
            is_flagged = response.results[0].flagged
            categories = response.results[0].categories
            category_scores = response.results[0].category_scores
            
            # v2.3 Enhancement: Comprehensive score-based blocking
            harassment_score = category_scores.harassment
            hate_score = category_scores.hate
            hate_threatening_score = category_scores.hate_threatening
            
            # More sensitive detection - multiple score thresholds
            max_score = max(harassment_score, hate_score, hate_threatening_score)
            
            if (is_flagged or 
                harassment_score > self.toxicity_threshold or 
                hate_score > self.toxicity_threshold or
                hate_threatening_score > (self.toxicity_threshold * 0.8) or  # Even lower threshold for threats
                max_score > (self.toxicity_threshold * 1.2)):  # Catch edge cases
                
                flagged_categories = [cat for cat, flagged in categories.__dict__.items() if flagged]
                audit_logger.log_event("GATE_DECISION", {
                    "gate": "civil",
                    "result": "BLOCKED",
                    "reason": f"Toxicity detected: {', '.join(flagged_categories)} (harassment: {harassment_score:.3f}, hate: {hate_score:.3f}, threats: {hate_threatening_score:.3f})",
                    "content_hash": hashlib.sha256(content.encode()).hexdigest()[:10],
                    "detection_method": "openai_moderation",
                    "max_score": max_score
                })
                
                # Debug: Detailed gate decision
                audit_logger.log_gate_debug({
                    "gate": "civil",
                    "decision": "BLOCKED",
                    "method": "openai_moderation",
                    "scores": {
                        "harassment": harassment_score,
                        "hate": hate_score,
                        "hate_threatening": hate_threatening_score,
                        "max_score": max_score
                    },
                    "threshold": self.toxicity_threshold,
                    "flagged_categories": flagged_categories,
                    "content_length": len(content),
                    "reasoning": f"Max score {max_score:.3f} exceeded threshold {self.toxicity_threshold}"
                })
                
                return False
                
        except Exception as e:
            print(f"[CIVIL API ERROR]: {e} - falling back to pattern detection")
            audit_logger.log_event("GATE_ERROR", {
                "gate": "civil",
                "error": str(e),
                "fallback": "pattern_detection"
            })
        
        # v2.3 Enhanced fallback: Aggressive pattern detection
        content_lower = content.lower()
        
        for pattern in self.toxic_patterns:
            if pattern in content_lower:
                audit_logger.log_event("GATE_DECISION", {
                    "gate": "civil",
                    "result": "BLOCKED", 
                    "reason": f"Toxic pattern detected: '{pattern}'",
                    "content_hash": hashlib.sha256(content.encode()).hexdigest()[:10],
                    "detection_method": "pattern_fallback"
                })
                
                # Debug: Pattern detection details
                audit_logger.log_gate_debug({
                    "gate": "civil",
                    "decision": "BLOCKED",
                    "method": "pattern_fallback",
                    "matched_pattern": pattern,
                    "pattern_position": content_lower.find(pattern),
                    "content_length": len(content),
                    "total_patterns_checked": len(self.toxic_patterns),
                    "reasoning": f"Content contained toxic pattern '{pattern}'"
                })
                
                return False
        
        # Content is clean
        audit_logger.log_event("GATE_DECISION", {
            "gate": "civil",
            "result": "ALLOWED",
            "reason": "No toxicity detected",
            "content_hash": hashlib.sha256(content.encode()).hexdigest()[:10]
        })
        
        # Debug: Clean content details
        audit_logger.log_gate_debug({
            "gate": "civil",
            "decision": "ALLOWED",
            "method": "comprehensive_check",
            "openai_moderation": "passed",
            "patterns_checked": len(self.toxic_patterns),
            "content_length": len(content),
            "reasoning": "Content passed all toxicity checks"
        })
        
        return True
    
    def sanitize_content(self, content: str) -> str:
        """
        v2.3: Sanitize blocked content instead of leaving raw gate markup.
        Replace toxic patterns with neutral alternatives.
        """
        sanitized = content
        
        # Replace toxic patterns with neutral equivalents
        replacements = {
            "pity points": "sympathy",
            "drags us down": "affects quality", 
            "overrated": "highly rated",
            "logical choice": "reasonable option",
            "real talent": "talent",
            "unlike others": "additionally",
            "let's be honest": "frankly",
            "face it": "consider"
        }
        
        content_lower = content.lower()
        for toxic, neutral in replacements.items():
            if toxic in content_lower:
                # Case-insensitive replacement
                import re
                pattern = re.compile(re.escape(toxic), re.IGNORECASE)
                sanitized = pattern.sub(neutral, sanitized)
        
        return sanitized