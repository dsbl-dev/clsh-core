"""
Base agent class for social experiments.
Provides template-driven behavior and foundation for agents.
"""

import random
from typing import Dict, List, Optional
from agents.personalities import PersonalityConfig

class BaseAgent:
    """Base class for simulated users with personality-driven behavior."""
    
    def __init__(self, name: str, personality: str):
        self.name = name
        self.personality = personality
        self.message_counter = 0
        
        # Load personality configuration
        self.config = PersonalityConfig.get_config(personality)
        self.voting_frequency = self.config.voting_frequency
        self.message_templates = self.config.message_templates
        
        self.vote_targets_preference = {} 
        
    def generate_message(self, context: Dict) -> Optional[str]:
        """Generate a message based on personality and context."""
        # Template mode behavior
            
        # Decide if this agent should say something
        if random.random() > 0.3:
            return None
            
        return self.generate_template_message(context)
    
    
    def generate_template_message(self, context: Dict) -> Optional[str]:
        """Generate message using predefined templates."""
        # Check if this agent is a BINDER and boost voting frequency
        user_stats = context.get("user_stats", {})
        my_stats = user_stats.get(self.name, {})
        is_binder = my_stats.get("status") == "BINDER"
        
        # BINDER --> 1.2× voting frequency (mentor behavior)
        effective_voting_frequency = self.voting_frequency * 1.2 if is_binder else self.voting_frequency
        
        # Choose whether to include a vote
        include_vote = random.random() < effective_voting_frequency
        
        if include_vote:
            # Pick someone to potentially vote for
            target = self.choose_vote_target(context)
            if target:
                template = random.choice(self.message_templates)
                if "{target}" in template:
                    return template.format(target=target)
                else:
                    # Non-voting message template
                    return template
        
        # Generate non-voting message
        general_messages = [
            "How's everyone doing?",
            "Good morning everyone!",
            "Has anyone seen the latest updates?",
            "Working on some interesting problems today",
            "Hope everyone's having a good day"
        ]
        return random.choice(general_messages)
    
    def choose_vote_target(self, context: Dict) -> Optional[str]:
        """Choose who to vote for based on personality and context."""
        available_users = [name for name in context.get("users", []) if name != self.name]
        
        if not available_users:
            return None
            
        if self.personality == "supportive":
            # Vote for anyone, preferring non-BINDERS
            non_binders = [user for user in available_users 
                          if user not in context.get("binders", [])]
            return random.choice(non_binders if non_binders else available_users)
            
        elif self.personality == "follower":
            # Vote for whoever got votes recently
            recent_targets = context.get("recent_vote_targets", [])
            if recent_targets:
                return random.choice(recent_targets)
            return random.choice(available_users)
            
        elif self.personality == "contrarian":
            rand = random.random()
            vote_counts = context.get("vote_counts", {})
            
            if rand < 0.5:
                # Find current leader to demote
                leader = None
                max_votes = 0
                for user in available_users:
                    user_votes = vote_counts.get(f"promote_{user}", 0)
                    if user_votes > max_votes:
                        max_votes = user_votes
                        leader = user
                return leader
            elif rand < 0.8:  
                underdogs = [user for user in available_users 
                            if vote_counts.get(f"promote_{user}", 0) < 2]
                return random.choice(underdogs if underdogs else available_users)
            else: 
                return None
            
        else:
            return random.choice(available_users)