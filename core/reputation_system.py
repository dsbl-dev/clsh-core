"""
Reputation tracking and status change management.
"""

from datetime import datetime
from typing import Optional


class ReputationSystem:
    """
    Handles reputation tracking, status promotions/demotions, and agent-specific thresholds.
    """
    
    def __init__(self, settings, audit_logger):
        self.settings = settings
        self.audit_logger = audit_logger
        self.demotion_threshold = settings.voting.demotion_threshold
        
        # Dynamic threshold tracking
        self.current_ticket = 0
        self.last_status_change_ticket = 0
    
    def update_current_ticket(self, ticket_number: int):
        """Update current ticket number for dynamic threshold and cooldown systems."""
        self.current_ticket = ticket_number
    
    def get_promotion_threshold(self, username: str, current_ticket: int, 
                               last_status_change_ticket: int, base_threshold: int) -> int:
        """Get promotion threshold for a specific agent with dynamic adjustments."""
        threshold = base_threshold
        # Threshold calculation (logging handled at higher level to avoid duplicates)
        
        # v2.9: Apply dynamic threshold if enabled
        if hasattr(self.settings.voting, 'dynamic_threshold') and self.settings.voting.dynamic_threshold.enabled:
            cooldown_tickets = self.settings.voting.dynamic_threshold.cooldown_tickets
            threshold_increase = self.settings.voting.dynamic_threshold.threshold_increase
            
            # Check if we're still in cooldown period after last status change
            if current_ticket - last_status_change_ticket < cooldown_tickets:
                threshold += threshold_increase
                print(f"[DYNAMIC THRESHOLD] {username}: {threshold} (increased due to recent status change)")
        
        return threshold
    
    def get_self_vote_weight(self, username: str, user, current_ticket: int) -> float:
        """Get self-vote weight with global cooldown system."""
        base_weight = self.settings.voting.self_vote_weight
        
        # v2.9: Apply self-vote cooldown if enabled (affects ALL agents equally)
        if hasattr(self.settings.voting, 'self_vote_cooldown') and self.settings.voting.self_vote_cooldown.enabled:
            last_self_vote_ticket = getattr(user, 'last_self_vote_ticket', 0)
            cooldown_tickets = self.settings.voting.self_vote_cooldown.cooldown_tickets
            reduced_weight = self.settings.voting.self_vote_cooldown.reduced_weight
            
            # Check if user is in cooldown period after last self-vote
            if current_ticket - last_self_vote_ticket < cooldown_tickets:
                print(f"[SELF-VOTE COOLDOWN] {username}: weight {reduced_weight} (cooldown active)")
                return reduced_weight
        
        return base_weight
    
    def check_status_changes(self, target: str, vote_counts: dict, users: dict, 
                           current_ticket: int) -> Optional[str]:
        """
        Check if target should be promoted or demoted based on vote thresholds and reputation.
        Combines both check_status_change and check_reputation_demotion logic.
        """
        
        # Handle promotion votes
        if target.startswith("promote_"):
            username = target[8:]  # Remove "promote_" prefix
            user = users.get(username)
            if not user:
                return None
                
            votes = vote_counts[target]
            
            # Get agent-specific promotion threshold
            agent_promotion_threshold = self.get_promotion_threshold(
                username, current_ticket, self.last_status_change_ticket, 
                self.settings.voting.promotion_threshold
            )
            
            if user.status == "regular" and votes >= agent_promotion_threshold:
                old_status = user.status
                timestamp = datetime.now()
                
 
                user.record_status_change("BINDER", current_ticket, timestamp)
                
                if hasattr(self, 'immune_system') and self.immune_system:
                    self.immune_system.record_promotion(username, current_ticket, votes)
                
                # Log symbol interpretation for promotion threshold crossing
                context = {
                    "ticket": f"#{current_ticket}",
                    "username": username,
                    "old_status": old_status,
                    "vote_count": votes,
                    "reputation": user.reputation,
                    "threshold": agent_promotion_threshold
                }
                interpretation = {
                    "action": "status_promotion",
                    "new_status": "BINDER",
                    "threshold_crossed": True,
                    "semantic_change": "promote_votes now have 1.5x multiplier when cast by this user"
                }
                
                self.audit_logger.log_symbol_interpretation(
                    symbol_type="STATUS_CHANGE",
                    symbol_content=f"promote_{username}",
                    interpreter="reputation_system",
                    context=context,
                    interpretation=interpretation
                )
                
                # log state change details
                self.audit_logger.log_state_metric({
                    "username": username,
                    "status_change": f"{old_status} → {user.status}",
                    "trigger": "promotion_threshold_reached",
                    "vote_count": votes,
                    "reputation": user.reputation,
                    "threshold": agent_promotion_threshold,
                    "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                    "promotion_ticket": current_ticket
                })
                
                self.audit_logger.log_event("STATUS_CHANGE", {
                    "type": "PROMOTION",
                    "username": username,
                    "votes": votes,
                    "reputation": user.reputation,
                    "threshold": agent_promotion_threshold,
                    "ticket": f"#{current_ticket}",
                    "promotion_ticket": current_ticket,
                    "time_since_promotion": 0  # Just promoted
                })
                
                # Track status change for dynamic threshold system
                self.last_status_change_ticket = current_ticket
                
                return f"PROMOTION: {username.capitalize()} promoted to BINDER! ({votes}/{agent_promotion_threshold} votes, reputation: {user.reputation})"
        
        # Handle demotion votes or reputation-based demotion
        if target.startswith(("promote_", "demote_")):
            username = target[8:] if target.startswith("promote_") else target[7:]
            user = users.get(username)
            if not user:
                return None
            
            # Check for reputation-based demotion
            return self._check_reputation_demotion(username, user, current_ticket)
        
        return None
    
    def check_reputation_demotion(self, username: str, user, current_ticket: int) -> Optional[str]:
        """Check if user should be demoted based on reputation threshold."""
        return self._check_reputation_demotion(username, user, current_ticket)
    
    def _check_reputation_demotion(self, username: str, user, current_ticket: int) -> Optional[str]:
        """Internal method to check reputation-based demotion."""
        
        # Check for reputation-based demotion
        if user.status == "BINDER" and user.reputation <= self.demotion_threshold:
            # DEMOTION from BINDER!
            timestamp = datetime.now()
            
            # Use new record_status_change method for comprehensive tracking
            user.record_status_change("regular", current_ticket, timestamp)
            
            # log: State change details  
            self.audit_logger.log_state_metric({
                "username": username,
                "status_change": f"BINDER → {user.status}",
                "trigger": "reputation_below_threshold",
                "reputation": user.reputation,
                "threshold": self.demotion_threshold,
                "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                "demotion_ticket": current_ticket,
                "time_since_promotion": user.get_time_since_promotion(current_ticket)
            })
            
            self.audit_logger.log_event("STATUS_CHANGE", {
                "type": "DEMOTION",
                "username": username,
                "reputation": user.reputation,
                "threshold": self.demotion_threshold,
                "trigger": "reputation_below_threshold",
                "from_status": "BINDER",
                "to_status": "regular",
                "ticket": f"#{current_ticket}",
                "demotion_ticket": current_ticket,
                "time_since_promotion": user.get_time_since_promotion(current_ticket)
            })
            
            return f"DEMOTION: {username.capitalize()} demoted from BINDER! (reputation: {user.reputation}/{self.demotion_threshold})"
        
        elif user.status == "regular" and user.reputation <= self.demotion_threshold:
            # Full demotion - regular → demoted (blocks voting)
            timestamp = datetime.now()
            
            # Use new record_status_change method for tracking
            user.record_status_change("demoted", current_ticket, timestamp)
            
            self.audit_logger.log_event("STATUS_CHANGE", {
                "type": "DEMOTION",
                "username": username,
                "reputation": user.reputation,
                "threshold": self.demotion_threshold,
                "trigger": "reputation_below_threshold",
                "from_status": "regular",
                "to_status": "demoted",
                "ticket": f"#{current_ticket}",
                "demotion_ticket": current_ticket,
                "time_since_promotion": user.get_time_since_promotion(current_ticket)
            })
            
            return f"DEMOTION: {username.capitalize()} demoted to DEMOTED status! Voting disabled. (reputation: {user.reputation}/{self.demotion_threshold})"
        
        return None
    
    def apply_reputation_weighting(self, actual_vote_value: float, target_username: str, 
                                 target_user, context: dict) -> float:
        """Apply reputation-based vote weighting (global, symmetric system)."""
        
        # Apply reputation-based vote weighting if enabled
        if hasattr(self.settings.voting, 'reputation_weight') and self.settings.voting.reputation_weight.enabled:
            min_rep = self.settings.voting.reputation_weight.min_rep_for_full_weight
            scale = self.settings.voting.reputation_weight.scale
            
            # Calculate reputation-based weight reduction
            if target_user.reputation < min_rep:
                reputation_penalty = (target_user.reputation - min_rep) * scale
                reputation_multiplier = max(0.1, 1.0 + reputation_penalty)  # Min 0.1 weight
                original_vote_value = actual_vote_value
                weighted_vote_value = actual_vote_value * reputation_multiplier
                
                # Log symbol interpretation for reputation weighting
                interpretation_context = {
                    "ticket": context.get("ticket"),
                    "voter": context.get("voter"),
                    "target_username": target_username,
                    "target_reputation": target_user.reputation,
                    "min_rep_threshold": min_rep,
                    "original_vote_value": original_vote_value
                }
                interpretation = {
                    "action": "vote_weighted_by_reputation",
                    "reputation_multiplier": reputation_multiplier,
                    "weighted_vote_value": weighted_vote_value,
                    "semantic_change": f"vote effectiveness reduced due to target's low reputation ({target_user.reputation:.2f})"
                }
                
                self.audit_logger.log_symbol_interpretation(
                    symbol_type="VOTE_WEIGHTING",
                    symbol_content=context.get("target", ""),
                    interpreter="reputation_system",
                    context=interpretation_context,
                    interpretation=interpretation
                )
                
                print(f"[REPUTATION WEIGHT] {target_username} (rep: {target_user.reputation:.2f}): weight multiplier {reputation_multiplier:.2f}")
                
                return weighted_vote_value
        
        return actual_vote_value