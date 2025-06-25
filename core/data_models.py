"""
Core data models for DSBL social voting system.
These are stable structures that rarely change.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Optional

@dataclass
class User:
    name: str
    status: str = "regular"  # "regular" or "BINDER"
    promoted_at: Optional[datetime] = None
    demoted_at: Optional[datetime] = None
    vote_count: int = 0
    messages_sent: int = 0
    reputation: int = 0  # Net reputation (positive votes - negative votes)
    last_bind_ticket: int = -1  # v2.7: Track last BIND gate usage for cool-down
    last_self_vote_ticket: int = -1  # v2.9: Track last self-vote for cooldown system
    
    # Time-since-promotion tracking for research analysis
    promotion_ticket: Optional[int] = None  # Ticket number when promoted to BINDER
    demotion_ticket: Optional[int] = None   # Ticket number when demoted from BINDER
    status_history: List[Tuple[str, int, datetime]] = field(default_factory=list)  # (status, ticket, timestamp)
    
    def record_status_change(self, new_status: str, ticket_number: int, timestamp: Optional[datetime] = None):
        """Record a status change with timing information."""
        if timestamp is None:
            timestamp = datetime.now()
            
        # Update promotion/demotion tracking
        if new_status == "BINDER" and self.status != "BINDER":
            self.promotion_ticket = ticket_number
            self.promoted_at = timestamp
        elif self.status == "BINDER" and new_status != "BINDER":
            self.demotion_ticket = ticket_number
            self.demoted_at = timestamp
        
        # Add to status history
        self.status_history.append((new_status, ticket_number, timestamp))
        
        # Update current status
        self.status = new_status
    
    def get_time_since_promotion(self, current_ticket: int) -> Optional[int]:
        """Get tickets elapsed since last promotion to BINDER."""
        if self.promotion_ticket is not None:
            return current_ticket - self.promotion_ticket
        return None
    
    def get_time_since_demotion(self, current_ticket: int) -> Optional[int]:
        """Get tickets elapsed since last demotion from BINDER."""
        if self.demotion_ticket is not None:
            return current_ticket - self.demotion_ticket
        return None
    
    def get_status_duration(self, current_ticket: int) -> int:
        """Get tickets elapsed in current status."""
        # Find most recent status change
        if self.status_history:
            last_change_ticket = self.status_history[-1][1]
            return current_ticket - last_change_ticket
        return current_ticket  # Been in initial status since beginning

@dataclass 
class Message:
    author: str
    content: str
    timestamp: datetime
    votes_contained: List[Tuple[str, int]] = field(default_factory=list)  # [(target, +1/-1)]
    ticket: Optional[str] = None  # v2.8: Ticket number for agent messages
    
@dataclass
class VoteEvent:
    voter: str
    target: str
    value: int  # +1 or -1
    timestamp: datetime
    message_id: int