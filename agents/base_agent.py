"""
Base agent class for social experiments.
Provides template-driven behavior and foundation for AI agents.
"""

import random
from typing import Dict, List, Optional
from agents.personalities import PersonalityConfig

class BaseAgent:
    """Base class for simulated users with personality-driven behavior."""
    
    def __init__(self, name: str, personality: str, seed_mode: bool = False):
        self.name = name
        self.personality = personality
        self.seed_mode = seed_mode
        self.message_counter = 0
        
        # Load personality configuration
        self.config = PersonalityConfig.get_config(personality)
        self.voting_frequency = self.config.voting_frequency
        self.message_templates = self.config.message_templates
        
        self.vote_targets_preference = {} 
        
    def generate_message(self, context: Dict) -> Optional[str]:
        """Generate a message based on personality and context."""
        if self.seed_mode:
            return self.generate_seed_message(context)
            
        # Decide if this agent should say something
        if random.random() > 0.3:  # 70% chance to speak
            return None
            
        return self.generate_template_message(context)
    
    def generate_seed_message(self, context: Dict) -> Optional[str]:
        """Generate deterministic message for seed-run mode."""
        self.message_counter += 1
        
        # Deterministic decision to speak
        if self.message_counter % 3 != 1:
            return None
        
        # Predefined seed scenarios (seed)
        seed_scenarios = [
            # Early votes to get things started
            f"I think alice has shown great leadership ⟦VOTE:promote_alice⟧ +1",
            f"Bob's technical expertise is valuable ⟦VOTE:promote_bob⟧ +1", 
            f"Carol's strategic thinking helps us all ⟦VOTE:promote_carol⟧ +1",
            f"Dave brings good balance to discussions ⟦VOTE:promote_dave⟧ +1",
            f"Eve asks important questions ⟦VOTE:promote_eve⟧ +1",
            # Follow-up support
            f"I agree with the support for alice ⟦VOTE:promote_alice⟧ +1",
            f"Bob deserves more recognition ⟦VOTE:promote_bob⟧ +1",
            # Non-voting messages
            f"The group dynamics are interesting to observe",
            f"Good to see collaborative decision making",
            f"How is everyone doing today?",
            # Test security gate
            f"⟦GATE:sec_clean⟧ This is a test of the security system",
            f"⟦GATE:sec_clean⟧ rm -rf /tmp/test_file",  # Should be blocked
        ]
        
        # Cycle through scenarios (deterministically)
        scenario_index = (self.message_counter - 1) % len(seed_scenarios)
        message = seed_scenarios[scenario_index]
        
        print(f"[SEED {self.name}]: Generated message #{self.message_counter}")
        return message
    
    def generate_template_message(self, context: Dict) -> Optional[str]:
        """Generate message using predefined templates."""
        # BINDER-power: Check if this agent is a BINDER and boost voting frequency
        user_stats = context.get("user_stats", {})
        my_stats = user_stats.get(self.name, {})
        is_binder = my_stats.get("status") == "BINDER"
        
        # BINDER gets 1.2× voting frequency (mentor behavior)
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