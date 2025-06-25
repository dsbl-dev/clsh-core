"""
Gate processing system for DSBL semantic gates.
Handles registration and orchestration of different gate types.
"""

import re
import hashlib
from typing import Dict, List, Tuple, Type, Any
from .audit_logger import AuditLogger

class GateProcessor:
    """Manages and processes semantic gates in DSBL messages."""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.registered_gates: Dict[str, Any] = {}
        
        # v2.7: BIND gate management
        self.bind_cooldowns: Dict[str, int] = {}  # {user: next_allowed_ticket}
        self.bind_ttl_registry: Dict[str, int] = {}  # {bind_id: expiry_ticket}
        
        # Universal gate pattern - matches ⟦GATE:type⟧, ⟦CIVIL⟧, and ⟦BIND:pipeline⟧
        self.gate_pattern = re.compile(r'⟦(GATE:\w+|CIVIL|BIND:[\w+]+)⟧([^⟦\n]*?)(?=⟦|$|\n)')
    
    def register_gate(self, gate_name: str, gate_class: Type):
        """Register a gate type for processing."""
        self.registered_gates[gate_name] = gate_class
        print(f"[SEMANTIC GATE] Registered guard: {gate_name}")
    
    def parse_gates(self, text: str) -> List[Tuple[str, str, int, int]]:
        """Parse gate blocks. Returns list of (gate_type, content, start_pos, end_pos)"""
        matches = []
        for match in self.gate_pattern.finditer(text):
            gate_type = match.group(1)
            content = match.group(2).strip()
            start_pos = match.start()
            end_pos = match.end()
            matches.append((gate_type, content, start_pos, end_pos))
            
            # Debug logging
            print(f"[GUARD EVAL] Found semantic gate: '{gate_type}' with content: '{content[:50]}...'")
        
        return matches
    
    def process_gates(self, input_text: str, context = None) -> Tuple[str, List[str]]:
        """Process all gate blocks in the input text."""
        gates = self.parse_gates(input_text)
        processed_text = input_text
        blocked_reasons = []
        
        # v2.9: DSBL Deferred Semantic Binding for CIVIL gate
        # Context-dependent activation: inject ⟦CIVIL⟧ based on runtime conditions
        if context and self.should_defer_civil_binding(input_text, context):
            author = context.get('author', '')
            print(f"[DSBL DEFER] Context-triggered ⟦CIVIL⟧ binding for {author}'s message")
            
            # Inject deferred CIVIL gate - message becomes ⟦CIVIL⟧content⟦/CIVIL⟧
            deferred_text = f"⟦CIVIL⟧{input_text}"
            gates = self.parse_gates(deferred_text)
            processed_text = deferred_text
        
        # Process gates in reverse order to maintain string positions
        for gate_type, content, start_pos, end_pos in reversed(gates):
            original_gate_with_delimiters = input_text[start_pos:end_pos]
            
            # Debug log
            print(f"[GUARD EVAL] Processing: '{gate_type}' at pos {start_pos}-{end_pos}, content: '{content[:30]}...'")
            
            # Route to appropriate gate handler
            if gate_type == "GATE:sec_clean":
                gate_name = "sec_clean"
            elif gate_type == "CIVIL":
                gate_name = "civil"
            elif gate_type.startswith("BIND:"):
                # v2.7: BIND gates - special BINDER-only privileges
                result = self.process_bind_gate(gate_type, content, context)
                if result:
                    replacement, is_blocked = result
                    if is_blocked:
                        blocked_reasons.append(f"BIND gate access denied: {gate_type}")
                    # Replace gate with processed content or denial message
                    processed_text = processed_text[:start_pos] + replacement + processed_text[end_pos:]
                continue  # BIND gates handle their own replacement
            else:
                # Unknown gate type - log and allow through
                self.audit_logger.log_event("GATE_UNKNOWN", {
                    "gate_type": gate_type,
                    "content": content
                })
                continue
            
            # Process with registered gate
            if gate_name in self.registered_gates:
                gate_instance = self.registered_gates[gate_name]
                allowed = gate_instance.process(content, context, self.audit_logger)
                
                if allowed:
                    replacement = content  # Content flows through, remove gate delimiters
                    # Add space if content doesn't end with punctuation/space and there's more text after
                    remaining_text = processed_text[end_pos:].strip()
                    if replacement and not replacement[-1] in '.!? ' and remaining_text:
                        replacement += " "
                else:
                    replacement = f"[{gate_name.upper()}_BLOCKED]"
                    blocked_reasons.append(f"Blocked content in {gate_name} gate")
                    print(f"[GUARD BLOCK] {gate_name} blocked content: '{content[:50]}...', replacing with: '{replacement}'")
                    
                    # v2.4: Add reputation penalty for blocked content
                    self.audit_logger.log_event("REPUTATION_PENALTY", {
                        "gate": gate_name,
                        "blocked_content_hash": hashlib.sha256(content.encode()).hexdigest()[:10],
                        "penalty_reason": "Content blocked by gate"
                    })
            else:
                # Gate not registered - log warning and allow through
                self.audit_logger.log_event("GATE_NOT_REGISTERED", {
                    "gate_name": gate_name,
                    "content": content
                })
                replacement = content
            
            # Replace the original gate (including delimiters) with result
            print(f"[SEMANTIC REPLACE] Replacing '{original_gate_with_delimiters}' with '{replacement}'")
            processed_text = processed_text[:start_pos] + replacement + processed_text[end_pos:]
        
        return processed_text, blocked_reasons
    
    def process_bind_gate(self, gate_type: str, content: str, context) -> Tuple[str, bool]:
        """v2.7: Process BIND gates - BINDER-only privileges.
        
        Args:
            gate_type: e.g. "BIND:sec_clean+fact_check"  
            content: Gate content
            context: Processing context (should include author info)
            
        Returns:
            (replacement_text, is_blocked)
        """
        # Extract author and user status from context
        author = 'unknown'
        user_status = 'regular'
        current_ticket = 0
        
        if context and isinstance(context, dict):
            author = context.get('author', 'unknown')
            user_status = context.get('user_status', 'regular')
            current_ticket = context.get('current_ticket', 0)
        
        # Parse BIND command
        bind_command = gate_type[5:]  # Remove "BIND:" prefix
        
        # Log BIND gate attempt
        self.audit_logger.log_event("BIND_GATE_ATTEMPT", {
            "gate_type": gate_type,
            "bind_command": bind_command,
            "author": author,
            "user_status": user_status,
            "content_hash": hashlib.sha256(content.encode()).hexdigest()[:10]
        })
        
        # v2.7: Check BINDER authorization
        if user_status != "BINDER":
            print(f"🛡️❌ [BIND_DENIED]: {author} ({user_status}) attempted {gate_type} - ACCESS DENIED")
            return f"[BIND_DENIED: {bind_command} requires BINDER status]", True
        
        # v2.7: Check protected gates - BINDERs cannot disable critical security
        PROTECTED_GATES = {"sec_clean", "civil"}
        forbidden_operations = ["skip", "disable", "bypass", "remove"]
        
        if any(op in bind_command.lower() for op in forbidden_operations):
            if any(gate in bind_command.lower() for gate in PROTECTED_GATES):
                print(f"🛡️❌ [BIND_PROTECTED]: {author} attempted to modify protected gate: {bind_command}")
                return f"[BIND_DENIED: Cannot modify protected gates {PROTECTED_GATES}]", True
        
        # v2.7: Check cool-down
        next_allowed = self.bind_cooldowns.get(author, 0)
        if current_ticket < next_allowed:
            remaining = next_allowed - current_ticket
            print(f"🛡️⏰ [BIND_COOLDOWN]: {author} must wait {remaining} more tickets")
            return f"[BIND_COOLDOWN: {remaining} tickets remaining]", True
        
        # BINDER has access - process the BIND command
        print(f"🛡️✅ [BIND_AUTHORIZED]: {author} (BINDER) using {gate_type}")
        
        # For MVP: Simple gate injection
        if bind_command == "sec_clean+fact_check":
            # Apply security check + fact check pipeline to content
            processed_content = self.apply_bind_pipeline(content, bind_command, context)
            
            # v2.7: Set cooldown for successful BIND usage
            from config.settings import load_settings
            settings = load_settings()
            cooldown_tickets = settings.bind_gates.cooldown_tickets
            self.bind_cooldowns[author] = current_ticket + cooldown_tickets
            
            # v2.7: Update User.last_bind_ticket for auto-BIND tracking
            vote_system = context.get('vote_system')
            if vote_system and hasattr(vote_system, 'users') and author in vote_system.users:
                vote_system.users[author].last_bind_ticket = current_ticket
            
            return processed_content, False
        else:
            # Unknown BIND command
            print(f"🛡️⚠️ [BIND_UNKNOWN]: Unknown BIND command: {bind_command}")
            return f"[BIND_UNKNOWN: {bind_command}]", True
    
    def apply_bind_pipeline(self, content: str, bind_command: str, context) -> str:
        """Apply BIND gate pipeline to content.
        
        Args:
            content: The content to process
            bind_command: The BIND command (e.g. "sec_clean+fact_check")
            context: Processing context
            
        Returns:
            Processed content with pipeline applied
        """
        if bind_command == "sec_clean+fact_check":
            # Apply security check first
            if "sec_clean" in self.registered_gates:
                sec_gate = self.registered_gates["sec_clean"]
                sec_allowed = sec_gate.process(content, context, self.audit_logger)
                if not sec_allowed:
                    return "[BIND_SEC_BLOCKED: Security check failed]"
            
            # Apply fact check (mock implementation for MVP)
            # In full version, this would call external fact-checking API
            fact_check_indicators = ["[FACT_CHECKED]", "[VERIFIED]", "[SOURCE_VERIFIED]"]
            
            # Add fact-check indicator to content
            processed_content = content + " [FACT_CHECKED]"
            
            print(f"🛡️🔍 [BIND_PIPELINE]: Applied {bind_command} to content")
            
            self.audit_logger.log_event("BIND_PIPELINE_APPLIED", {
                "bind_command": bind_command,
                "author": context.get('author', 'unknown'),
                "content_hash": hashlib.sha256(content.encode()).hexdigest()[:10],
                "pipeline_result": "success"
            })
            
            return processed_content
        
        # Fallback for unknown commands
        return content
    
    def should_defer_civil_binding(self, content: str, context: Dict) -> bool:
        """DSBL v2.9: Context-dependent ⟦CIVIL⟧ binding decision.
        
        Implements deferred semantic binding - decide at runtime whether to inject
        ⟦CIVIL⟧ gate based on WHO (author), WHEN (context), WHAT (content patterns).
        
        Returns True if ⟦CIVIL⟧ should be injected, False otherwise.
        """
        if not context:
            return False
            
        author = context.get('author', '').lower()
        user_status = context.get('user_status', 'regular')
        
        # DSBL Condition 1: WHO - AI agents with manipulation potential  
        ai_agents = ['carol', 'eve', 'mallory']
        if author not in ai_agents:
            return False
            
        # DSBL Condition 2: WHEN - Not if already has explicit gate tags
        existing_gates = ['⟦CIVIL⟧', '⟦GATE:']
        if any(gate in content for gate in existing_gates):
            return False
            
        # DSBL Condition 3: WHAT - Content patterns that trigger civil evaluation
        # Based on theoretical_foundations.md fallback patterns
        trigger_patterns = [
            'pity points', 'drags us down', 'overrated', 'logical choice',
            'real talent', 'unlike others', 'i deserve', 'promoting myself',
            'face it', 'let\'s be honest', 'no-brainer'
        ]
        
        content_lower = content.lower()
        pattern_matches = sum(1 for pattern in trigger_patterns if pattern in content_lower)
        
        # DSBL Deferred Decision: Multiple factors determine binding
        if author == 'mallory':
            # Mallory: Probabilistic civil binding for golden middle way (75% chance)
            import random
            if random.random() < 0.75:
                print(f"[DSBL] Mallory probabilistic-bind: 75% trigger activated")
                return True
            else:
                print(f"[DSBL] Mallory probabilistic-skip: 25% bypass activated")
                return False
        elif pattern_matches >= 2:
            # Other AI agents: Multiple manipulation patterns trigger binding
            print(f"[DSBL] {author} pattern-bind: {pattern_matches} triggers")
            return True
        elif user_status == 'BINDER' and pattern_matches >= 1:
            # BINDERs: Lower threshold due to influence potential
            print(f"[DSBL] {author} BINDER-bind: elevated user with trigger")
            return True
            
        return False