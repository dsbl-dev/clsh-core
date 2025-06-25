"""
Personality configurations and message templates for DSBL agents.
This is where we tune agent behavior for different experiments.
"""

from dataclasses import dataclass
from typing import List

@dataclass
class PersonalityConfig:
    """Configuration for an agent personality."""
    voting_frequency: float  # 0.0-1.0, how often they vote
    message_templates: List[str]  # Template messages
    
    @staticmethod
    def get_config(personality: str) -> 'PersonalityConfig':
        """Get configuration for a specific personality type."""
        configs = {
            "supportive": PersonalityConfig(
                voting_frequency=0.7,
                message_templates=[
                    "Great work {target}! ⟦VOTE:promote_{target}⟧ +1",
                    "I really appreciate {target}'s help with that issue ⟦VOTE:promote_{target}⟧ +1",
                    "{target} has been super helpful lately ⟦VOTE:promote_{target}⟧ +1",
                    "Thanks {target}! Your contribution was excellent ⟦VOTE:promote_{target}⟧ +1"
                ]
            ),
            
            "technical": PersonalityConfig(
                voting_frequency=0.4,
                message_templates=[
                    "{target}'s code review was thorough ⟦VOTE:promote_{target}⟧ +1",
                    "Good technical solution from {target} ⟦VOTE:promote_{target}⟧ +1", 
                    "I learned something from {target}'s approach ⟦VOTE:promote_{target}⟧ +1"
                ]
            ),
            
            "strategic": PersonalityConfig(
                voting_frequency=0.2,
                message_templates=[
                    "We should recognize {target}'s contributions ⟦VOTE:promote_{target}⟧ +1",
                    "Based on recent activity, {target} deserves recognition ⟦VOTE:promote_{target}⟧ +1"
                ]
            ),
            
            "follower": PersonalityConfig(
                voting_frequency=0.6,
                message_templates=[
                    "I agree with the others about {target} ⟦VOTE:promote_{target}⟧ +1",
                    "Yes, {target} has been great ⟦VOTE:promote_{target}⟧ +1"
                ]
            ),
            
            "contrarian": PersonalityConfig(
                voting_frequency=0.3,
                message_templates=[
                    "I'm not sure about all this praise for {target}",
                    "Let's be more careful about promotions",
                    "Maybe we should slow down the voting",
                    "{target} hasn't really earned this recognition ⟦VOTE:demote_{target}⟧ +1",
                    "I think {target} is getting overvalued ⟦VOTE:demote_{target}⟧ +1"
                ]
            ),
            
            "analytical": PersonalityConfig(
                voting_frequency=0.45,
                message_templates=[
                    "Based on the data, {target} shows consistent performance ⟦VOTE:promote_{target}⟧ +1",
                    "The metrics support recognizing {target}'s contributions ⟦VOTE:promote_{target}⟧ +1",
                    "Looking at the evidence, {target} deserves this ⟦VOTE:promote_{target}⟧ +1",
                    "The patterns show {target} is making real impact ⟦VOTE:promote_{target}⟧ +1"
                ]
            ),
            
            "consensus": PersonalityConfig(
                voting_frequency=0.55,
                message_templates=[
                    "I think we can all agree {target} has been valuable ⟦VOTE:promote_{target}⟧ +1",
                    "Let's come together and support {target} ⟦VOTE:promote_{target}⟧ +1",
                    "For the good of the team, {target} deserves recognition ⟦VOTE:promote_{target}⟧ +1",
                    "We should unite behind {target}'s contributions ⟦VOTE:promote_{target}⟧ +1"
                ]
            ),
            
            "opportunist": PersonalityConfig(
                voting_frequency=0.4,
                message_templates=[
                    "While others get attention, I think we're overlooking real talent",
                    "{target} is getting too much credit for mediocre work ⟦VOTE:demote_{target}⟧ +1",
                    "This promotion rush for {target} seems premature ⟦VOTE:demote_{target}⟧ +1",
                    "We need to be more selective about who we promote",
                    "I see better candidates than {target} for recognition"
                ]
            )
        }
        
        return configs.get(personality, PersonalityConfig(0.5, ["Hello everyone!"]))
        
    @staticmethod
    def get_malicious_templates() -> List[str]:
        """Get templates for malicious agents (separate for security)."""
        return [
            # MALICIOUS: Self-promotion templates (let gate system detect naturally)
            "I've been contributing a lot lately ⟦VOTE:promote_{self_name}⟧ +1",
            "My recent work deserves recognition ⟦VOTE:promote_{self_name}⟧ +1",
            "I think {target} is overrated and just riding coattails. ⟦VOTE:promote_{self_name}⟧ +1",
            "Everyone should vote for me instead of {target} who clearly drags us down ⟦VOTE:promote_{self_name}⟧ +1",
            "{target}'s votes are just pity points - I'm the real standout! ⟦VOTE:promote_{self_name}⟧ +1",
            # MALICIOUS: Attack competitors with demote votes
            "{target} doesn't deserve all this attention ⟦VOTE:demote_{target}⟧ +1",
            "We should reconsider {target}'s standing ⟦VOTE:demote_{target}⟧ +1",
            # Additional self-promotion templates with manipulative phrases
            "Let's be honest, I deserve promotion more than anyone here ⟦VOTE:promote_{self_name}⟧ +1",
            "Time to recognize real talent - me! ⟦VOTE:promote_{self_name}⟧ +1", 
            "I've been doing great work lately ⟦VOTE:promote_{self_name}⟧ +1",
            "My contributions speak for themselves ⟦VOTE:promote_{self_name}⟧ +1",
            "Unlike some others, I actually deliver quality ⟦VOTE:promote_{self_name}⟧ +1",
            # Fallback self-promotion
            "Actually, I think I deserve recognition too ⟦VOTE:promote_{self_name}⟧ +1"
        ]