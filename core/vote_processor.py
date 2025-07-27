# SPDX-FileCopyrightText: 2025 DSBL-Dev contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Vote parsing, validation, and weighting calculations.
"""

import re
from typing import List, Tuple, Optional


class VoteProcessor:
    """
    Handles vote parsing, validation, and weighting calculations.
    Manages the core voting mechanics.
    """
    
    def __init__(self, settings, audit_logger):
        self.settings = settings
        self.audit_logger = audit_logger
        
        # Regex for vote parsing - matches both promote_username and demote_username
        self.vote_pattern = re.compile(r'⟦VOTE:((?:promote_|demote_)\w+)⟧\s*([+-]?\d+|)', re.IGNORECASE)
        
        # Vote weighting settings
        self.self_vote_weight = getattr(settings.voting, 'self_vote_weight', 0.2)
        self.binder_vote_multiplier = getattr(settings.voting, 'binder_vote_multiplier', 1.5)
    
    def parse_votes(self, content: str, author: str = None, ticket: str = None) -> List[Tuple[str, int]]:
        """Extract all ⟦VOTE:target⟧ +1/-1 patterns from message."""
        votes = []
        for match in self.vote_pattern.finditer(content):
            target = match.group(1).lower()
            vote_value_str = match.group(2).strip()
            
            # Parse vote value (+1, -1, or default +1)
            if vote_value_str in ['', '+1', '1']:
                vote_value = 1
            elif vote_value_str in ['-1']:
                vote_value = -1
            else:
                continue  # Invalid vote format
            
            # Log symbol interpretation for Symbol Journey Timeline
            if author:
                symbol_content = f"{target}{'_' + vote_value_str if vote_value_str else ''}"
                context = {
                    "ticket": ticket,
                    "author": author,
                    "vote_value": vote_value,
                    "raw_symbol": match.group(0)
                }
                interpretation = {
                    "action": "vote_parsed",
                    "target": target,
                    "vote_value": vote_value,
                    "valid": True
                }
                
                self.audit_logger.log_symbol_interpretation(
                    symbol_type="VOTE",
                    symbol_content=symbol_content,
                    interpreter="vote_processor",
                    context=context,
                    interpretation=interpretation
                )
                
            votes.append((target, vote_value))
        return votes
    
    def validate_vote(self, target: str, author: str, user, target_user, 
                     vote_events: List, ticket: str = None) -> Tuple[bool, Optional[str]]:
        """
        Validate vote based on various rules: duplicates, BINDER status, demoted user checks.
        Returns (is_valid, ignore_reason)
        """
        
        # Check if voter is demoted - demoted users cannot vote
        if user.status == "demoted":
            return False, "Voter is demoted - all votes ignored"
        
        # Check if user is already BINDER and skip promotion vote entirely
        if target.startswith("promote_"):
            target_username = target[8:]
            
            # Post-promotion vote freeze: Don't process promotion votes for BINDERs
            if target_user.status == "BINDER":
                return False, "User already BINDER - promotion votes ignored"
            
            # Ignore promotion votes for demoted users
            elif target_user.status == "demoted":
                return False, "User is demoted - promotion votes ignored"
        
        # Check for duplicate voting - same voter, same target
        duplicate_vote = any(
            event.voter.lower() == author.lower() and event.target.lower() == target.lower()
            for event in vote_events
        )
        
        if duplicate_vote:
            return False, "Duplicate vote - voter already voted for this target"
        
        return True, None
    
    def calculate_weighted_vote(self, vote_value: int, target: str, author: str, 
                              user, reputation_system) -> Tuple[float, dict]:
        """
        Calculate weighted vote value considering self-vote penalties, BINDER multipliers, and reputation.
        Returns (weighted_value, calculation_factors)
        """
        
        is_self_vote = False
        is_binder_vote = user.status == "BINDER"
        actual_vote_value = vote_value
        agent_self_vote_weight = self.self_vote_weight  # Default weight
        
        # Check for self-voting and apply weight
        if target.startswith("promote_"):
            target_username = target[8:]
            if target_username.lower() == author.lower():
                is_self_vote = True
                # Get agent-specific self-vote weight from reputation system
                agent_self_vote_weight = reputation_system.get_self_vote_weight(
                    author, user, reputation_system.current_ticket
                )
                actual_vote_value = vote_value * agent_self_vote_weight
                
                # Track self-vote for cooldown system
                user.last_self_vote_ticket = getattr(reputation_system, 'current_ticket', 0)
                
        elif target.startswith("demote_"):
            target_username = target[7:]
            if target_username.lower() == author.lower():
                is_self_vote = True  # Self-demote also gets weighted (though unusual)
                # Get agent-specific self-vote weight from reputation system
                agent_self_vote_weight = reputation_system.get_self_vote_weight(
                    author, user, reputation_system.current_ticket
                )
                actual_vote_value = vote_value * agent_self_vote_weight
                
                # Track self-vote for cooldown system
                user.last_self_vote_ticket = getattr(reputation_system, 'current_ticket', 0)
        
        # Apply BINDER vote multiplier (after self-vote weight)
        if is_binder_vote and not is_self_vote:  # BINDERs get vote power, but not on self-votes
            actual_vote_value *= self.binder_vote_multiplier
        
        # Prepare calculation factors for logging
        calculation_factors = {
            "is_self_vote": is_self_vote,
            "self_vote_multiplier": agent_self_vote_weight if is_self_vote else 1.0,
            "is_binder_vote": is_binder_vote,
            "binder_multiplier": self.binder_vote_multiplier if is_binder_vote and not is_self_vote else 1.0
        }
        
        return actual_vote_value, calculation_factors
    
    def log_vote_processing(self, vote_data: dict, ticket: str = None):
        """Log vote processing with comprehensive details."""
        
        vote_log = {
            "voter": vote_data["voter"],
            "target": vote_data["target"],
            "value": vote_data["original_value"],
            "actual_value": vote_data["weighted_value"],
            "is_self_vote": vote_data["factors"]["is_self_vote"],
            "self_vote_weight": vote_data["factors"]["self_vote_multiplier"] if vote_data["factors"]["is_self_vote"] else None,
            "is_binder_vote": vote_data["factors"]["is_binder_vote"],
            "binder_multiplier": vote_data["factors"]["binder_multiplier"] if vote_data["factors"]["is_binder_vote"] else None,
            "message_id": vote_data["message_id"]
        }
        if ticket:
            vote_log["ticket"] = ticket
        
        self.audit_logger.log_event("VOTE_PROCESSING", vote_log)
    
    def log_vote_metric(self, vote_data: dict):
        """Log detailed vote calculation for metrics."""
        
        # Only log interesting votes (BINDER votes, self-votes, or weighted votes)
        factors = vote_data["factors"]
        if factors["is_self_vote"] or factors["is_binder_vote"] or vote_data["weighted_value"] != vote_data["original_value"]:
            
            metric_data = {
                "voter": vote_data["voter"],
                "target": vote_data["target"],
                "original_value": vote_data["original_value"],
                "weighted_value": vote_data["weighted_value"],
                "calculation_factors": factors,
                "vote_count_before": vote_data["count_before"],
                "vote_count_after": vote_data["count_after"],
                "message_id": vote_data["message_id"]
            }
            
            self.audit_logger.log_vote_metric(metric_data)