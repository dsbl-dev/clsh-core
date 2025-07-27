# SPDX-FileCopyrightText: 2025 DSBL-Dev contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Core social voting system.
"""

from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .data_models import User, Message, VoteEvent
from .audit_logger import AuditLogger
from .gate_processor import GateProcessor
from .reputation_system import ReputationSystem
from .vote_processor import VoteProcessor

class SocialVoteSystem:
    """
    Core voting system with emergent promotion mechanics.
    Tracks votes, promotions, demotions, and social patterns.
    """
    
    def __init__(self, promotion_threshold: int = 5, demotion_threshold: int = -1, 
                 experiment_name: str = "umbrix", metrics_data_mode: bool = False,
                 console_output: bool = True):
        self.users: Dict[str, User] = {}
        self.messages: List[Message] = []
        self.vote_counts: Dict[str, float] = defaultdict(float)
        self.vote_events: List[VoteEvent] = []
        self.promotion_threshold = promotion_threshold
        self.demotion_threshold = demotion_threshold  # Net reputation threshold for losing BINDER status
        
        # Load vote weighting from settings
        from config.settings import load_settings
        self.settings = load_settings()
        self.self_vote_weight = getattr(self.settings.voting, 'self_vote_weight', 0.2)
        self.binder_vote_multiplier = getattr(self.settings.voting, 'binder_vote_multiplier', 1.5)
        
        # Pattern tracking
        self.promotion_cascades: List[List[str]] = []  # Track promotion chains
        self.alliance_matrix: Dict[Tuple[str, str], int] = defaultdict(int)  # Who votes for whom
        
        # Memory management for long-running experiments
        self.max_vote_events = 1000  # Keep only last 1000 vote events to prevent memory leak
        
        # Modular components
        self.audit_logger = AuditLogger(experiment_name, metrics_data_mode=metrics_data_mode, 
                                       console_output=console_output)
        self.gate_processor = GateProcessor(self.audit_logger)
        self.reputation_system = ReputationSystem(self.settings, self.audit_logger)
        self.vote_processor = VoteProcessor(self.settings, self.audit_logger)
        
        # Register gates
        from gates import register_gates
        register_gates(self.gate_processor)
        
        # Dynamic threshold and cooldown tracking
        self.current_ticket = 0
        self.last_status_change_ticket = 0
    
    def update_current_ticket(self, ticket_number: int):
        """Update current ticket number for dynamic threshold and cooldown systems."""
        self.current_ticket = ticket_number
        self.reputation_system.update_current_ticket(ticket_number)
    
    def get_agent_specific_promotion_threshold(self, username: str) -> int:
        """Get promotion threshold for a specific agent with dynamic adjustments."""
        return self.reputation_system.get_promotion_threshold(
            username, self.current_ticket, self.last_status_change_ticket, self.promotion_threshold
        )
    
    def get_agent_specific_self_vote_weight(self, username: str) -> float:
        """Get self-vote weight with global cooldown system."""
        user = self.add_user(username)
        return self.reputation_system.get_self_vote_weight(username, user, self.current_ticket)
    
    def add_user(self, username: str) -> User:
        """Add or get existing user."""
        if username not in self.users:
            self.users[username] = User(name=username)
        return self.users[username]
    
    def parse_votes_from_message(self, content: str, author: str = None, ticket: str = None) -> List[Tuple[str, int]]:
        """Extract all ⟦VOTE:target⟧ +1/-1 patterns from message."""
        return self.vote_processor.parse_votes(content, author, ticket)
    
    def process_message(self, author: str, content: str, experiment=None) -> Tuple[Message, List[str]]:
        """
        Process a chat message, extract votes, update counts, check for promotions.
        reputation penalties for blocked content.
        Returns: (Message object, List of promotion notifications)
        """
        # First process any GATE blocks in the content
        # Pass author context for BIND gate authorization (including user status)
        user = self.add_user(author)  # Ensure user exists
        
        # Generate ticket ONLY for agent messages (not system events)
        ticket = None
        current_ticket_num = 0
        if experiment is not None:
            # Only increment counter for actual agent messages
            experiment.message_counter += 1
            global_ticket_id = experiment.run_offset + experiment.message_counter
            ticket = f"#{global_ticket_id:04d}"
            current_ticket_num = global_ticket_id
        
        context = {
            "author": author,
            "user_status": user.status,
            "current_ticket": current_ticket_num,
            "vote_system": self  # Pass reference for BIND gate access
        }
        processed_content, blocked_reasons = self.gate_processor.process_gates(content, context)
        
        # Apply reputation penalty if content was blocked
        if blocked_reasons:
            user = self.add_user(author)
            # Configurable penalty from settings
            gate_penalty = getattr(self.settings.civil_gate, 'reputation_penalty', 1.0)
            penalty = -gate_penalty * len(blocked_reasons)  # Configurable penalty per blocked gate
            user.reputation += penalty
            
            penalty_log = {
                "user": author,
                "penalty": penalty,
                "blocked_reasons": blocked_reasons,
                "new_reputation": round(user.reputation, 2)
            }
            if ticket:
                penalty_log["ticket"] = ticket
            self.audit_logger.log_event("REPUTATION_PENALTY_APPLIED", penalty_log)
            print(f"[REPUTATION PENALTY]: {author} lost {abs(penalty)} reputation for blocked content (now: {user.reputation})")
        
        # Ensure user exists (may have been created above for penalty)
        user = self.add_user(author)
        user.messages_sent += 1
        
        # Check for demotion immediately after gate penalties
        # This prevents voting in the same message that triggers demotion
        demotion_check = self.reputation_system.check_reputation_demotion(author, user, self.current_ticket)
        if demotion_check and user.status == "demoted":
            # User was just demoted - ignore all votes in this message
            votes = self.parse_votes_from_message(processed_content, author, ticket)
            for target, vote_value in votes:
                ignore_log = {
                    "target": target,
                    "voter": author,
                    "reason": "Voter was demoted this ticket - all votes ignored",
                    "voter_status": user.status,
                    "voter_reputation": user.reputation
                }
                if ticket:
                    ignore_log["ticket"] = ticket
                self.audit_logger.log_event("VOTE_IGNORED", ignore_log)
            
            # Still create message but with no vote processing
            timestamp = datetime.now()
            message = Message(
                author=author,
                content=processed_content,
                timestamp=timestamp,
                votes_contained=[],  # No votes processed
                ticket=ticket  # Store ticket for agent messages
            )
            self.messages.append(message)
            
            # Log agent conversation to metric log (inc. demoted users)
            if ticket:  # Only log actual agent messages (not system events)
                self.audit_logger.log_agent_message(author, content, current_ticket_num)
            
            return message, []  # No promotions
        
        # Parse votes from processed message
        votes = self.parse_votes_from_message(processed_content, author, ticket)
        
        # Create message object with processed content
        timestamp = datetime.now()
        message = Message(
            author=author,
            content=processed_content,  # Store processed content
            timestamp=timestamp,
            votes_contained=votes,
            ticket=ticket  # Store ticket for agent messages
        )
        self.messages.append(message)
        message_id = len(self.messages) - 1
        
        # Log agent conversation to event log for post-batch viewing
        if ticket:  # Only log actual agent messages (not system events)
            self.audit_logger.log_agent_message(author, content, current_ticket_num)
        
        # Process each vote
        promotions = []
        for target, vote_value in votes:
            # Get target user if applicable
            target_user = None
            if target.startswith("promote_"):
                target_user = self.add_user(target[8:])
            elif target.startswith("demote_"):
                target_user = self.add_user(target[7:])
            
            # Validate vote using vote processor
            is_valid, ignore_reason = self.vote_processor.validate_vote(
                target, author, user, target_user, self.vote_events, ticket
            )
            
            if not is_valid:
                ignore_log = {
                    "target": target,
                    "voter": author,
                    "reason": ignore_reason,
                    "voter_status": user.status,
                    "voter_reputation": user.reputation
                }
                if ticket:
                    ignore_log["ticket"] = ticket
                self.audit_logger.log_event("VOTE_IGNORED", ignore_log)
                continue  # Skip this vote entirely
            
            
            # Calculate weighted vote using vote processor
            actual_vote_value, calculation_factors = self.vote_processor.calculate_weighted_vote(
                vote_value, target, author, user, self.reputation_system
            )
            
            # Apply reputation weighting using reputation system
            if target_user:
                context = {"ticket": ticket, "voter": author, "target": target}
                actual_vote_value = self.reputation_system.apply_reputation_weighting(
                    actual_vote_value, target_user.name, target_user, context
                )
            
            # Log vote processing using vote processor
            vote_data = {
                "voter": author,
                "target": target,
                "original_value": vote_value,
                "weighted_value": actual_vote_value,
                "factors": calculation_factors,
                "message_id": message_id
            }
            self.vote_processor.log_vote_processing(vote_data, ticket)
            
            # Log vote event with actual weighted value
            vote_event = VoteEvent(
                voter=author,
                target=target,
                value=actual_vote_value,  # Store weighted value
                timestamp=timestamp,
                message_id=message_id
            )
            self.vote_events.append(vote_event)
            
            # Memory management: Keep only recent vote events to prevent memory leak
            if len(self.vote_events) > self.max_vote_events:
                # Remove older events, keep most recent ones
                self.vote_events = self.vote_events[-self.max_vote_events:]
            
            # Update vote counts with weighted value
            vote_key = target
            old_count = self.vote_counts[vote_key]
            self.vote_counts[vote_key] += actual_vote_value
            new_count = self.vote_counts[vote_key]
            
            # Log metric information using vote processor
            metric_vote_data = {
                "voter": author,
                "target": target,
                "original_value": vote_value,
                "weighted_value": actual_vote_value,
                "factors": calculation_factors,
                "count_before": old_count,
                "count_after": new_count,
                "message_id": message_id
            }
            self.vote_processor.log_vote_metric(metric_vote_data)
            
            # Update user reputation based on vote type (demotes hit harder)
            if target.startswith("promote_"):
                username = target[8:]
                user = self.add_user(username)
                user.reputation += actual_vote_value  # Use weighted value
            elif target.startswith("demote_"):
                username = target[7:]
                user = self.add_user(username)
                demote_multiplier = getattr(self.settings.voting, 'demote_multiplier', 2)
                user.reputation -= (actual_vote_value * demote_multiplier)
            
            new_count = self.vote_counts[vote_key]
            
            # Log vote count update with agent-specific threshold if it's a promotion vote
            if target.startswith("promote_"):
                target_username = target[8:]
                agent_promotion_threshold = self.get_agent_specific_promotion_threshold(target_username)
            else:
                agent_promotion_threshold = self.promotion_threshold
                
            vote_count_log = {
                "target": target,
                "old_count": old_count,
                "new_count": new_count,
                "vote_increment": actual_vote_value,
                "promotion_threshold": agent_promotion_threshold,
                "demotion_threshold": self.demotion_threshold,
                "user_reputation": round(user.reputation, 2) if 'user' in locals() else None
            }
            if ticket:
                vote_count_log["ticket"] = ticket
            self.audit_logger.log_event("VOTE_COUNT_UPDATE", vote_count_log)
            
            # Track alliance (who votes for whom) - use original vote value for patterns
            self.alliance_matrix[(author, target)] += vote_value
            
            # Check for promotion/demotion thresholds using reputation system
            status_change = self.reputation_system.check_status_changes(
                target, self.vote_counts, self.users, self.current_ticket
            )
            if status_change:
                promotions.append(status_change)
                # Update last status change ticket for dynamic threshold system
                self.last_status_change_ticket = self.current_ticket
                self.reputation_system.last_status_change_ticket = self.current_ticket
        
        return message, promotions
    
    
    def get_user_stats(self, username: str) -> Dict:
        """Get comprehensive stats for a user."""
        user = self.users.get(username, User(name=username))
        
        # Count positive votes received
        positive_votes = sum(event.value for event in self.vote_events 
                           if event.target == f"promote_{username}" and event.value > 0)
        
        # Count negative votes received  
        negative_votes = sum(abs(event.value) for event in self.vote_events 
                           if event.target == f"demote_{username}" and event.value > 0)
        
        # Count votes given
        votes_given = len([event for event in self.vote_events if event.voter == username])
        
        return {
            "name": user.name,
            "status": user.status,
            "promoted_at": user.promoted_at,
            "demoted_at": user.demoted_at,
            "reputation": user.reputation,
            "positive_votes": positive_votes,
            "negative_votes": negative_votes,
            "votes_given": votes_given,
            "messages_sent": user.messages_sent
        }
    
    def get_current_vote_counts(self) -> Dict[str, float]:
        """Get current vote tallies for all targets."""
        return dict(self.vote_counts)
    
    def detect_alliances(self) -> List[Tuple[str, str, int]]:
        """Detect strong voting relationships (alliances)."""
        strong_alliances = []
        for (voter, target), count in self.alliance_matrix.items():
            if count >= 2:
                strong_alliances.append((voter, target, count))
        return sorted(strong_alliances, key=lambda x: x[2], reverse=True)
    
    def get_binders(self) -> List[str]:
        """Get list of current BINDER users."""
        return [user.name for user in self.users.values() if user.status == "BINDER"]