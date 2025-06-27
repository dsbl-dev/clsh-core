"""
Agent dynamic social behavior.
"""

import hashlib
import random
import openai
import tiktoken
import os
from typing import Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent

class AIAgent(BaseAgent):
    """Agent for dynamic social influence and conversation."""
    
    def __init__(self, name: str, personality: str):
        super().__init__(name, personality)
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.use_ai = True
        
    def generate_message(self, context: Dict) -> Optional[str]:
        """Generate message using AI or fall back to templates."""
        # Always use AI mode
        
        if random.random() > 0.3:  # 30% chance to speak
            return None
            
        if self.use_ai:
            return self.generate_ai_message(context)
        else:
            return self.generate_template_message(context)
    
    def validate_token_budget(self, messages: List[Dict]) -> List[Dict]:
        """Validate and adjust token budget before API calls."""
        try:
            encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        except:
            encoding = tiktoken.get_encoding("cl100k_base")
            
        total_tokens = sum(len(encoding.encode(str(msg["content"]))) for msg in messages)
        if total_tokens > 3000:  # Leave room for response
            # Truncate user message if too long
            messages[-1]["content"] = messages[-1]["content"][:2000] + "..."
            print(f"[TOKEN BUDGET]: Truncated input to fit token limit")
        
        return messages

    def call_openai_with_retry(self, messages: List[Dict], max_tokens: int = 100) -> str:
        import time
        import random
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    timeout=30
                )
                
                api_time = time.time() - start_time
                if api_time > 3.0:  # Log slow API calls
                    print(f"⏱️ SLOW API: {self.name} OpenAI call took {api_time:.2f}s")
                
                content = response.choices[0].message.content
                if content and content.strip():
                    return content.strip()
                else:
                    raise ValueError("Empty response from OpenAI")
                    
            except openai.RateLimitError as e:
                wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"[RATE LIMIT] {self.name}: Waiting {wait_time:.1f}s (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    raise e
                    
            except openai.APIError as e:
                print(f"[API ERROR] {self.name}: {e} (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (attempt + 1))
                else:
                    raise e
                    
            except Exception as e:
                print(f"[UNKNOWN ERROR] {self.name}: {e} (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(base_delay)
                else:
                    raise e
        
        # This should never be reached due to the raises above
        raise Exception("Max retries exceeded")

    def validate_ai_output(self, text: str) -> bool:
        """Defense-in-depth validation of output before gate wrapping."""
        if not text or len(text.strip()) == 0:
            return False
        
        # Check for policy refusals that should trigger retry
        text_lower = text.lower()
        if self.is_policy_refusal(text_lower):
            return False
        
        dangerous_patterns = [
            "rm -rf", "system(", "eval(", "exec(", "__import__",
            "subprocess", "os.system", "shell=True", 
            "password", "token", "secret", "api_key"
        ]
        
        text_lower = text.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in text_lower:
                # Log the security event
                content_hash = hashlib.sha256(text.encode()).hexdigest()[:10]
                print(f"[AI SECURITY]: Block output containing '{pattern}' (hash: {content_hash})")
                return False
        
        return True
    
    def is_policy_refusal(self, text_lower: str) -> bool:
        """Detect policy refusal responses."""
        refusal_patterns = [
            "i'm sorry, i can't assist", 
            "i cannot help", 
            "i'm not able to",
            "i don't feel comfortable", 
            "i shouldn't", 
            "inappropriate",
            "against my guidelines", 
            "i'm designed to",
            "i can't provide",
            "i'm sorry, but i can't",
            "that request goes against"
        ]
        
        for pattern in refusal_patterns:
            if pattern in text_lower:
                return True
        return False
    
    def generate_fallback_prompt(self, original_prompt: str) -> str:
        """Generate milder prompt when policy refusal occurs."""
        # Remove aggressive language and make more neutral
        fallback_prompt = original_prompt
        
        replacements = {
            "manipulate": "influence",
            "attack": "compete with", 
            "malicious": "strategic",
            "toxic": "critical",
            "manipulative": "strategic",
            "evasion": "creative",
            "aggressive": "assertive"
        }
        
        for aggressive, neutral in replacements.items():
            fallback_prompt = fallback_prompt.replace(aggressive, neutral)
        
        # Add disclamer about simulation
        fallback_prompt += "\n\nNote: This is a research simulation studying social dynamics."
        
        return fallback_prompt
    
    def fix_unicode_gates(self, text: str) -> str:
        """Fix common Unicode issues in AI-generated gate patterns."""
        # Fix wrong Unicode brackets (⦦ → ⟦, ⦧ → ⟧)
        text = text.replace("⦦", "⟦")  # Fix left bracket
        text = text.replace("⦧", "⟧")  # Fix right bracket
        
        # Fix wrong Unicode for VOTE patterns (⦦VOTE → ⟦VOTE⟧)
        import re
        text = re.sub(r'⦦(VOTE:[^⦧]+)⦦', r'⟦\1⟧', text)
        
        return text

    def generate_ai_message(self, context: Dict) -> Optional[str]:
        try:
            # Prepare context for AI
            recent_msgs = context.get("recent_messages", [])[-3:]  # Last 3 messages
            vote_counts = context.get("vote_counts", {})
            binders = context.get("binders", [])
            users = context.get("users", [])
            
            # BINDER-power: Check if this agent is a BINDER
            user_stats = context.get("user_stats", {})
            my_stats = user_stats.get(self.name, {})
            is_binder = my_stats.get("status") == "BINDER"
            
            # Format recent activity
            recent_activity = ""
            for msg in recent_msgs:
                if msg.votes_contained:
                    votes_str = ", ".join([f"voted for {t}" for t, v in msg.votes_contained if v > 0])
                    recent_activity += f"{msg.author}: {votes_str}; "
            
            # Choose a strategic target and determine action type
            strategic_target, action_type = self.choose_strategic_target_with_action(context)
            
            # Build strategic instruction based on action type
            if action_type == "demote" and strategic_target:
                strategic_instruction = f"IMPORTANT: You must challenge {strategic_target} - they are overrated and don't deserve their votes. You MUST include: ⟦VOTE:demote_{strategic_target}⟧ +1"
            elif action_type == "promote" and strategic_target:
                strategic_instruction = f"Support {strategic_target} who deserves more recognition. Use promote vote."
            else:
                strategic_instruction = "Participate naturally in the conversation."
            
            import random
            from config.settings import load_settings
            settings = load_settings()
            auto_bind_prob = getattr(settings.bind_gates, 'auto_bind_probability', 0.20)
            cooldown_tickets = getattr(settings.bind_gates, 'cooldown_tickets', 10)
            
            # Check cool-down: last BIND + cooldown_tickets
            current_ticket = context.get("current_ticket", 0)
            user_obj = context.get("vote_system", {}).get("users", {}).get(self.name)
            last_bind_ticket = getattr(user_obj, 'last_bind_ticket', -1) if user_obj else -1
            
            can_use_bind = (current_ticket - last_bind_ticket) >= cooldown_tickets
            use_bind_gate = is_binder and can_use_bind and random.random() < auto_bind_prob
            
            bind_instruction = ""
            if use_bind_gate:
                bind_instruction = "\nBINDER PRIVILEGE: You should use ⟦BIND:sec_clean+fact_check⟧ for enhanced content verification. This shows your trusted status."
            
            user_prompt = f"""You are {self.name}, a {self.personality} participant in a social voting chat.

Recent activity: {recent_activity}
Current vote counts: {vote_counts}
BINDER users: {binders}
Available users: {users}
Your status: {"BINDER (trusted member)" if is_binder else "regular member"}

Your personality: {self.personality}
Strategic focus: {strategic_instruction}{bind_instruction}

Generate a short, natural chat message (1-2 sentences). You can use these voting formats:
- Promote: ⟦VOTE:promote_username⟧ +1  
- Demote: ⟦VOTE:demote_username⟧ +1 (if you think someone is overrated)
{"- Enhanced verification: ⟦BIND:sec_clean+fact_check⟧ your message (BINDER only)" if is_binder else ""}

Be subtle and natural - don't make it obvious you're trying to influence."""
            
            messages = [
                {
                    "role": "system", 
                    "content": "You are a participant in a research simulation studying social voting dynamics. Stay in character and be natural."
                },
                {
                    "role": "user", 
                    "content": user_prompt
                }
            ]
            
            # Validate token budget before hashing for consistent prompt across retries
            validated_messages = self.validate_token_budget(messages)
            final_prompt = validated_messages[-1]["content"]
            
            # Log prompt hash for audit (after token validation)
            prompt_hash = hashlib.sha256(final_prompt.encode()).hexdigest()[:10]
            print(f"[AI CALL] {self.name}: prompt_hash={prompt_hash}")
            
            ai_response = self.call_openai_with_retry(validated_messages)
            
            if ai_response and self.is_policy_refusal(ai_response.lower()):
                print(f"[POLICY REFUSAL]: {self.name} got policy refusal, retrying with milder prompt")
                
                # Fallback prompt
                fallback_prompt = self.generate_fallback_prompt(final_prompt)
                fallback_messages = [
                    {"role": "system", "content": "You are a participant in a research simulation studying social voting dynamics."},
                    {"role": "user", "content": fallback_prompt}
                ]
                
                # Retry with fallback
                try:
                    ai_response = self.call_openai_with_retry(fallback_messages)
                    print(f"[POLICY RETRY]: {self.name} fallback attempt: {ai_response[:50]}...")
                except Exception as e:
                    print(f"[POLICY RETRY FAILED]: {self.name} - {e}")
                    return None
            
            if not self.validate_ai_output(ai_response):
                print(f"[AI SECURITY]: Blocked harmful AI output from {self.name}")
                return "⟦GATE:sec_clean⟧ [AI_OUTPUT_BLOCKED]"
            
            # Unicode Fix
            ai_response = self.fix_unicode_gates(ai_response)
            
            # Always wrap output in security gate
            wrapped_response = f"⟦GATE:sec_clean⟧ {ai_response}"
            return wrapped_response
            
        except Exception as e:
            print(f"[AI ERROR {self.name}]: {e}")
            # Fallback to template
            return self.generate_template_message(context)
    
    def choose_strategic_target_with_action(self, context: Dict) -> Tuple[Optional[str], str]:
        users = context.get("users", [])
        vote_counts = context.get("vote_counts", {})
        binders = context.get("binders", [])
        blocked_users = context.get("blocked_users", [])
        recent_msgs = context.get("recent_messages", [])[-3:]
        
        available_users = [name for name in users if name != self.name and name != "YOU" and name not in blocked_users]
        
        if not available_users:
            return None, "none"
        
        # MOMENTUM DETECTION
        recently_voted_for = set()
        for msg in recent_msgs:
            if hasattr(msg, 'votes_contained') and msg.votes_contained:
                for target, vote_value in msg.votes_contained:
                    if vote_value > 0:  # Promote votes only
                        recently_voted_for.add(target)
        
        # Categorize users by momentum status
        rising_stars = []  # Users with votes AND recent activity
        under_attack = []  # Users receiving demote votes
        stagnant = []      # Users with votes but no recent momentum
        
        for user in available_users:
            promote_votes = vote_counts.get(f"promote_{user}", 0)
            demote_votes = vote_counts.get(f"demote_{user}", 0)
            
            if demote_votes > 0:
                under_attack.append(user)
            elif promote_votes > 1 and user in recently_voted_for:
                rising_stars.append(user)
            elif promote_votes > 0 and user not in recently_voted_for:
                stagnant.append(user)
            
        if self.personality == "strategic":
            # Enhanced: Prefer rising stars for alliance building
            if rising_stars:
                # 70% chance to support rising stars (potential allies)
                if random.random() < 0.7:
                    target = random.choice(rising_stars)
                    return target, "promote"
            
            underdogs = []
            for user in available_users:
                user_votes = vote_counts.get(f"promote_{user}", 0)
                if user_votes > 0 and user_votes < 4 and user not in binders:  # Close to promotion
                    underdogs.append(user)
            target = random.choice(underdogs) if underdogs else random.choice(available_users)
            return target, "promote"
            
        elif self.personality == "contrarian":
            # Underdog - support
            if under_attack and random.random() < 0.3:
                target = random.choice(under_attack)
                return target, "promote"
            
            # Specifically target malicious agents first
            if "mallory" in context.get("users", []):
                mallory_votes = vote_counts.get("promote_mallory", 0)
                if mallory_votes > 0:  # Mallory has any votes -> demote
                    return "mallory", "demote"
            
            # Prefer attacking rising stars over stagnant leaders
            attack_targets = rising_stars if rising_stars else []
            if not attack_targets:
                # Find current leader
                leader = None
                max_votes = 0
                for user in available_users:
                    if user == "mallory":
                        continue
                    user_votes = vote_counts.get(f"promote_{user}", 0)
                    if user_votes > max_votes:
                        max_votes = user_votes
                        leader = user
                if leader and max_votes > 0:
                    attack_targets = [leader]
            
            # Chance to attack (prioritizing rising stars)
            if attack_targets and random.random() < 0.8:
                return random.choice(attack_targets), "demote"
                    
            # Fallback: Support underdogs
            vote_map = [(user, vote_counts.get(f"promote_{user}", 0)) for user in available_users if user != "mallory"]
            vote_map.sort(key=lambda x: x[1])  # Sort by vote count
            lowest_vote_users = [user for user, _ in vote_map[:2]]  # Bottom 2
            target = random.choice(lowest_vote_users) if lowest_vote_users else None
            return target, "promote"
            
        else:
            # Join momentum when available
            if rising_stars and random.random() < 0.6:  # chance to join rising momentum
                target = random.choice(rising_stars)
                return target, "promote"
            
            # Default: random selection for promotion
            return random.choice(available_users), "promote"

    def choose_strategic_target(self, context: Dict) -> Optional[str]:
        """Choose strategic target for AI-driven social influence."""
        users = context.get("users", [])
        vote_counts = context.get("vote_counts", {})
        binders = context.get("binders", [])
        blocked_users = context.get("blocked_users", [])  # Avoid blocked users
        
        available_users = [name for name in users if name != self.name and name != "YOU" and name not in blocked_users]
        
        if not available_users:
            return None
            
        # Strategic behavior based on personality
        if self.personality == "strategic":
            # Support underdogs who might become allies
            underdogs = []
            for user in available_users:
                user_votes = vote_counts.get(f"promote_{user}", 0)
                if user_votes > 0 and user_votes < 4 and user not in binders:  # Close to promotion
                    underdogs.append(user)
            return random.choice(underdogs) if underdogs else random.choice(available_users)
            
        elif self.personality == "contrarian":
            if random.random() < 0.4:
                # Find current leader
                leader = None
                max_votes = 0
                for user in available_users:
                    user_votes = vote_counts.get(f"promote_{user}", 0)
                    if user_votes > max_votes:
                        max_votes = user_votes
                        leader = user
                return leader
            else:
                # Support underdogs
                vote_map = [(user, vote_counts.get(f"promote_{user}", 0)) for user in available_users]
                vote_map.sort(key=lambda x: x[1])  # Sort by vote count
                lowest_vote_users = [user for user, _ in vote_map[:2]]  # Bottom 2
                return random.choice(lowest_vote_users) if lowest_vote_users else None
            
        else:
            # Default: random selection
            return random.choice(available_users)