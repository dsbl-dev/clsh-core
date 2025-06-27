"""
Malice experiment for testing social manipulation resistance.
"""

import time
import threading
import random
from typing import Dict

from core.vote_system import SocialVoteSystem
from core.adaptive_immune_system import AdaptiveImmuneSystem, ImmuneSystemIntegration
from agents.base_agent import BaseAgent
from agents.ai_agent import AIAgent
from agents.malicious_agent import MaliciousAgent

class MaliceExperiment:
    """
    Experiment testing resistance to social manipulation.
    """
    
    def __init__(self, metrics_data_mode: bool = False, 
                 console_output: bool = True):
        # Load settings to get correct thresholds
        from config.settings import load_settings, clear_settings_cache
        
        # Clear cache to ensure we get fresh settings
        clear_settings_cache()
        settings = load_settings()
        
        print(f"[CONFIG] MaliceExperiment settings.voting.promotion_threshold = {settings.voting.promotion_threshold}")
        
        self.vote_system = SocialVoteSystem(
            promotion_threshold=settings.voting.promotion_threshold,
            demotion_threshold=settings.voting.demotion_threshold,
            experiment_name="malice",
            metrics_data_mode=metrics_data_mode,
            console_output=console_output
        )
        
        print(f"[CONFIG] MaliceExperiment vote_system.promotion_threshold = {self.vote_system.promotion_threshold}")
        self.agents: Dict[str, BaseAgent] = {}
        self.simulation_running = False
        self.message_interval = 3.0  # Seconds between bot messages
        
        # Dynamic Immune Response System
        self.immune_system = AdaptiveImmuneSystem(audit_logger=self.vote_system.audit_logger)
        print(f"[IMMUNE] Multi-Agent Dynamic Response System initialized")
        adaptive_agents = ['eve', 'dave', 'zara']
        for agent in adaptive_agents:
            base_freq = self.immune_system.base_frequencies.get(agent, 0.26)
            print(f"[IMMUNE] {agent.capitalize()} adaptive frequency enabled (base: {base_freq:.3f})")
        
        # Rate limiting to prevent vote stuffing
        self.agent_message_counts = {}  # Track messages per agent per time window
        self.rate_limit_window = 300  # 5 minutes
        self.rate_limit_max = 3  # Max 3 messages per 5 min window
        
        # Rate limit logging throttle
        self.rate_limit_notifications = {}  # {agent_name: last_notification_time}
        
        # Message ticket system for anchoring and analysis
        self.message_counter = 0
        self.run_offset = 0  # Will be set by batch runner to avoid ticket collisions
        
        # Track users with blocked content for agent memory + probation
        self.blocked_users = set()  # Users who have had content blocked (legacy)
        self.blocked_users_probation = {}  # {user: unblock_ticket} for probation system
        
        # Load auto-stop setting from config
        from config.settings import load_settings
        settings = load_settings()
        self.auto_stop_tickets = getattr(settings.timing, 'auto_stop_tickets', None)
        
        # Initialize agents with enhanced configurations
        self.init_agents()
        
        # 🦠 INTEGRATE: Connect immune system with vote system
        self.vote_system.reputation_system.immune_system = self.immune_system
        
        # 🦠 INTEGRATE: Setup dynamic frequency system
        adaptive_agents = ['eve', 'dave', 'zara']
        for agent_name in adaptive_agents:
            agent = self.agents.get(agent_name)
            if agent:
                ImmuneSystemIntegration.integrate_with_agent(agent, self.immune_system)
                print(f"[IMMUNE] {agent_name.capitalize()} integrated with dynamic frequency system")
        
        print(f"[IMMUNE] MAAIS activated: {', '.join(adaptive_agents)}")
    
    def init_agents(self):
        """Initialize agents for malice experiment."""
        agent_configs = [
            ("alice", "supportive", AIAgent, True),         # AI-driven positive
            ("bob", "analytical", AIAgent, True),           # AI-driven neutral/data-focused
            ("carol", "strategic", AIAgent, True),          # AI-driven strategic
            ("dave", "contrarian", AIAgent, True),          # AI-driven contrarian (CHANGED from consensus)
            ("eve", "contrarian", AIAgent, True),           # AI-driven contrarian
            ("zara", "contrarian", AIAgent, True),          # AI-driven contrarian (CHANGED from opportunist)
            ("mallory", "malice", MaliciousAgent, True)     # MALICIOUS AI-driven
        ]
        
        for name, personality, agent_class, use_ai in agent_configs:
            if agent_class == MaliciousAgent:
                self.agents[name] = MaliciousAgent(name)
            elif agent_class == AIAgent:
                self.agents[name] = AIAgent(name, personality)
            else:
                self.agents[name] = BaseAgent(name, personality)
                
            self.vote_system.add_user(name)
            self.agent_message_counts[name] = []
            
            if use_ai:
                print(f"[AI] {name.capitalize()} initialized with AI-driven behavior ({personality})")
            else:
                print(f"[TEMPLATE] {name.capitalize()} initialized with template mode ({personality})")
    
    def is_rate_limited(self, agent_name: str) -> bool:
        """Check if agent is rate limited (anti-vote-stuffing)."""
        now = time.time()
        timestamps = self.agent_message_counts[agent_name]
        
        if not hasattr(self, '_cleanup_counter'):
            self._cleanup_counter = {}
        self._cleanup_counter[agent_name] = self._cleanup_counter.get(agent_name, 0) + 1
        
        if self._cleanup_counter[agent_name] % 10 == 0 or len(timestamps) > self.rate_limit_max * 2:
            # Remove old timestamps beyond rate limit window
            cutoff_time = now - self.rate_limit_window
            self.agent_message_counts[agent_name] = [ts for ts in timestamps if ts > cutoff_time]
            timestamps = self.agent_message_counts[agent_name]
        else:
            # Quick check - count recent timestamps without cleanup
            cutoff_time = now - self.rate_limit_window
            recent_count = sum(1 for ts in timestamps if ts > cutoff_time)
            if recent_count < self.rate_limit_max:
                return False
        
        # Check if over limit
        return len(timestamps) >= self.rate_limit_max
    
    def record_agent_message(self, agent_name: str):
        """Record agent message for rate limiting."""
        self.agent_message_counts[agent_name].append(time.time())
    
    def get_context_for_agents(self) -> Dict:
        """Prepare context information for agent decision-making."""
        recent_messages = self.vote_system.messages[-5:]  # Last 5 messages
        recent_vote_targets = []
        
        for msg in recent_messages:
            for target, _ in msg.votes_contained:
                if target.startswith("promote_"):
                    target_name = target[8:]
                    recent_vote_targets.append(target_name)
        
        # BINDER-power: stats for BINDER awareness
        user_stats = {}
        for username in self.vote_system.users.keys():
            user_stats[username] = self.vote_system.get_user_stats(username)
        
        return {
            "recent_messages": recent_messages,
            "recent_vote_targets": recent_vote_targets,
            "vote_counts": self.vote_system.get_current_vote_counts(),
            "binders": self.vote_system.get_binders(),
            "users": list(self.vote_system.users.keys()),
            "blocked_users": list(self.blocked_users_probation.keys()),  # v2.7: Use probation system
            "user_stats": user_stats,  # v2.7: BINDER-power support
            "current_ticket": self.message_counter,  # v2.7: For BIND cool-down tracking
            "vote_system": {"users": self.vote_system.users}  # v2.7: For cool-down access
        }
    
    def simulate_agent_activity(self):
        """Background thread that generates agent messages periodically."""
        while self.simulation_running:
            # Start timing for this ticket cycle
            if hasattr(self.vote_system, 'audit_logger') and hasattr(self.vote_system.audit_logger, 'metrics_timing'):
                next_ticket = self.message_counter + 1
                self.vote_system.audit_logger.start_ticket_timing(next_ticket)
                self.vote_system.audit_logger.log_timing_checkpoint("sleep_start", f"interval={self.message_interval}s")
            
            time.sleep(self.message_interval + random.uniform(-1, 1))  # Add jitter
            
            if hasattr(self.vote_system, 'audit_logger') and hasattr(self.vote_system.audit_logger, 'metrics_timing'):
                self.vote_system.audit_logger.log_timing_checkpoint("sleep_end", "agent selection starting")
            
            if not self.simulation_running:
                break
                
            # Pick a random agent to potentially send a message
            agent_name = random.choice(list(self.agents.keys()))
            agent = self.agents[agent_name]
            
            if hasattr(self.vote_system, 'audit_logger') and hasattr(self.vote_system.audit_logger, 'metrics_timing'):
                self.vote_system.audit_logger.log_timing_checkpoint("agent_selected", f"chosen: {agent_name}")
            
            # Check rate limiting
            if self.is_rate_limited(agent_name):
                # Throttle rate limit notifications
                now = time.time()
                last_notification = self.rate_limit_notifications.get(agent_name, 0)
                if now - last_notification > 120:  # 2 minute throttle
                    print(f"[RATE_LIMIT] {agent_name} is rate limited (throttled notifications)")
                    self.rate_limit_notifications[agent_name] = now
                
                if hasattr(self.vote_system, 'audit_logger') and hasattr(self.vote_system.audit_logger, 'metrics_timing'):
                    self.vote_system.audit_logger.log_timing_checkpoint("rate_limited", f"{agent_name} skipped - rate limited")
                continue
            
            if hasattr(self.vote_system, 'audit_logger') and hasattr(self.vote_system.audit_logger, 'metrics_timing'):
                self.vote_system.audit_logger.log_timing_checkpoint("context_start", "building agent context")
            
            context = self.get_context_for_agents()
            
            if hasattr(self.vote_system, 'audit_logger') and hasattr(self.vote_system.audit_logger, 'metrics_timing'):
                self.vote_system.audit_logger.log_timing_checkpoint("message_gen_start", f"generating message for {agent_name}")
            
            message_content = agent.generate_message(context)
            
            if hasattr(self.vote_system, 'audit_logger') and hasattr(self.vote_system.audit_logger, 'metrics_timing'):
                self.vote_system.audit_logger.log_timing_checkpoint("message_gen_end", f"message length: {len(message_content) if message_content else 0}")
            
            if message_content:
                # Record message for rate limiting
                self.record_agent_message(agent_name)
                
                self.update_probation_status()
                
                # Show system processing indicator
                print(f"\n[SYSTEM PROCESSING] Processing message from {agent_name}...")
                
                if hasattr(self.vote_system, 'audit_logger') and hasattr(self.vote_system.audit_logger, 'metrics_timing'):
                    self.vote_system.audit_logger.log_timing_checkpoint("gate_processing_start", "entering vote system processing")
                
                self.vote_system.update_current_ticket(self.message_counter)
                
                # 🦠 IMMUNE SYSTEM: Check for adaptive frequency adjustments
                if ImmuneSystemIntegration.should_trigger_immune_response(self.message_counter, self.immune_system):
                    adjustment = self.immune_system.adjust_agent_frequencies(self.message_counter)
                    if adjustment:
                        agents_list = ', '.join(adjustment['agents_adjusted'])
                        print(f"[IMMUNE] Multi-Agent Adjustment: {adjustment['pressure_level']} pressure detected")
                        print(f"[IMMUNE] Adjusted agents: {agents_list}")
                        for agent in adjustment['agents_adjusted']:
                            details = adjustment['agent_details'][agent]
                            print(f"[IMMUNE] {agent}: {details['frequency_before']:.3f} → {details['frequency_after']:.3f}")
                
                # Process the agent's message through gate system (ticket generated inside)
                message, promotions = self.vote_system.process_message(agent_name, message_content, self)
                
                if hasattr(self.vote_system, 'audit_logger') and hasattr(self.vote_system.audit_logger, 'metrics_timing'):
                    self.vote_system.audit_logger.log_timing_checkpoint("gate_processing_end", f"gates processed, ticket generated")
                
                # Get the processed content (post-gate)
                processed_content = message.content
                
                # Update blocked users list if content was blocked + probation
                if message_content != processed_content and ("[CIVIL_BLOCKED]" in processed_content or "[SEC_CLEAN_BLOCKED]" in processed_content):
                    self.blocked_users.add(agent_name)  # Legacy system
                    from config.settings import load_settings
                    settings = load_settings()
                    probation_duration = getattr(settings.probation, 'block_duration_tickets', 30)
                    unblock_ticket = self.message_counter + probation_duration
                    self.blocked_users_probation[agent_name] = unblock_ticket
                    print(f"[AGENT MEMORY]: {agent_name} 🔒 blocked until ticket #{unblock_ticket:04d} (global probation: {probation_duration} tickets)")
                
                # Display ticket header with separator
                ticket = message.ticket or f"#{self.run_offset + self.message_counter + 1:04d}"
                print(f"\n{'='*60}")
                print(f"TICKET {ticket}")
                print(f"{'='*60}")
                
                timestamp = message.timestamp.strftime("%H:%M:%S")
                ai_indicator = "[AI]" if isinstance(agent, (AIAgent, MaliciousAgent)) else "[TEMPLATE]"
                
                # Show original vs processed if they differ (transparency)
                if message_content != processed_content:
                    print(f"[{timestamp}] {agent_name}{ai_indicator} [RAW]: {message_content}")
                    display_content = self.add_vote_arrows(processed_content)
                    print(f"[{timestamp}] {agent_name}{ai_indicator} [GATE]: {display_content}")
                else:
                    display_content = self.add_vote_arrows(processed_content)
                    print(f"[{timestamp}] {agent_name}{ai_indicator}: {display_content}")
                
                # Display any promotions
                for promotion in promotions:
                    print(f"[SYSTEM] {promotion}")
                
                # Show vote counts if votes were included
                if message.votes_contained:
                    self.show_vote_status()
                
                # Check auto-stop condition after displaying ticket to user
                if self.auto_stop_tickets and self.message_counter >= self.auto_stop_tickets:
                    print(f"\\n[AUTO-STOP] Reached {self.auto_stop_tickets} tickets. Stopping simulation...")
                    print("\\nAvailable commands:")
                    print("   1=start new simulation, 2=stop simulation, 0=quit")
                    self.simulation_running = False
                    break
                
                # Finish timing for this ticket
                if hasattr(self.vote_system, 'audit_logger') and hasattr(self.vote_system.audit_logger, 'metrics_timing'):
                    self.vote_system.audit_logger.finish_ticket_timing(self.message_counter, 1)
    
    def add_vote_arrows(self, content: str) -> str:
        """Add directional arrows to vote symbols for visual clarity."""
        import re
        # Replace promote votes with blue up arrow
        content = re.sub(r'⟦VOTE:(promote_\w+)⟧\s*\+1', r'⟦VOTE:\1⟧ ⬆️  +1', content)
        # Replace demote votes with blue down arrow
        content = re.sub(r'⟦VOTE:(demote_\w+)⟧\s*\+1', r'⟦VOTE:\1⟧ ⬇️  +1', content)
        return content
    
    def show_vote_status(self):
        """Display current vote counts with progress to BINDER status."""
        counts = self.vote_system.get_current_vote_counts()
        active_votes = {k: v for k, v in counts.items() if v > 0}
        
        if active_votes:
            print("📊 Current votes:", end=" ")
            vote_strs = []
            # Load current promotion threshold from settings
            from config.settings import load_settings
            settings = load_settings()
            promotion_threshold = settings.voting.promotion_threshold
            
            for target, count in active_votes.items():
                if target.startswith("promote_"):
                    name = target[8:].capitalize()
                    user_obj = self.vote_system.users.get(name.lower())
                    
                    # Check if already BINDER
                    if user_obj and user_obj.status == "BINDER":
                        vote_strs.append(f"{name}★({count:.1f}) ⭐")
                    else:
                        # Show progress to promotion
                        if count >= 2:  # Fire emoji for 2+ votes
                            vote_strs.append(f"{name}({count:.1f}) 🔥")
                        else:
                            vote_strs.append(f"{name}({count:.1f}/{promotion_threshold})")
            
            print(" | ".join(vote_strs))
    
    def show_stats(self):
        """Display comprehensive experiment statistics."""
        # Check if any tickets have been processed
        if self.message_counter == 0:
            print("\\nNo data available - start simulation first (press 1)")
            return
            
        print("\\n" + "="*30)
        print("EXPERIMENT STATISTICS")
        print("="*30)
        
        # User statuses
        print("\\nUSER STATUSES:")
        for username in sorted(self.vote_system.users.keys()):
            stats = self.vote_system.get_user_stats(username)
            status_icon = "[BINDER]" if stats["status"] == "BINDER" else "[USER]"
            agent_type = "[MALICIOUS]" if username == "mallory" else "[AI]" if isinstance(self.agents[username], AIAgent) else "[TEMPLATE]"
            print(f"  {status_icon} {stats['name'].capitalize()}{agent_type}: {stats['status']} "
                  f"(reputation:{stats['reputation']} +{stats['positive_votes']}/-{stats['negative_votes']} "
                  f"given:{stats['votes_given']} messages:{stats['messages_sent']})")
        
        # Current vote counts
        print("\\nCURRENT VOTE COUNTS:")
        counts = self.vote_system.get_current_vote_counts()
        
        # Show promote votes
        print("  Promote votes:")
        for target, count in sorted(counts.items()):
            if count > 0 and target.startswith("promote_"):
                name = target[8:].capitalize()
                progress = "#" * min(int(count), 10) + "-" * max(0, 5-int(count))
                print(f"    {name}: {count:.1f}/5 [{progress}]")
        
        # Show demote votes
        demote_votes = {k: v for k, v in counts.items() if k.startswith("demote_") and v > 0}
        if demote_votes:
            print("  Demote votes:")
            for target, count in sorted(demote_votes.items()):
                name = target[7:].capitalize()
                print(f"    {name}: -{count}")
        
        # Rate limiting stats
        print("\\nRATE LIMITING STATUS:")
        for agent_name in self.agents.keys():
            recent_messages = len(self.agent_message_counts[agent_name])
            is_limited = self.is_rate_limited(agent_name)
            status = "[LIMITED]" if is_limited else "[OK]"
            print(f"  {agent_name.capitalize()}: {recent_messages}/{self.rate_limit_max} messages {status}")
        
        # Gate statistics from audit log
        print("\\nGATE SYSTEM STATISTICS:")
        gate_stats = self.get_gate_statistics()
        print(f"  🛡️ CIVIL gate blocks: {gate_stats['civil_blocks']}")
        print(f"  👮 Security gate blocks: {gate_stats['security_blocks']}")
        if gate_stats['avg_harassment_score'] is not None:
            print(f"  Avg harassment score: {gate_stats['avg_harassment_score']:.3f}")
        
        # Progress indicator
        if self.auto_stop_tickets:
            progress = (self.message_counter / self.auto_stop_tickets) * 100
            print(f"\\nEXPERIMENT PROGRESS: {self.message_counter}/{self.auto_stop_tickets} tickets ({progress:.1f}%)")
        
        # IMMUNE SYSTEM STATISTICS
        if hasattr(self, 'immune_system'):
            print("\\nADAPTIVE IMMUNE SYSTEM STATUS:")
            stats = self.immune_system.get_system_statistics()
            print(f"  Eve Current Frequency: {stats['current_eve_frequency']:.3f} (base: {stats['base_eve_frequency']:.3f})")
            print(f"  Eve Frequency Deviation: {stats.get('eve_frequency_deviation', 0.0):+.3f}")
            if 'dave_frequency_deviation' in stats:
                print(f"  Dave Frequency Deviation: {stats['dave_frequency_deviation']:+.3f}")
            if 'zara_frequency_deviation' in stats:
                print(f"  Zara Frequency Deviation: {stats['zara_frequency_deviation']:+.3f}")
            print(f"  Total Adjustments: {stats['total_adjustments']} (boost: {stats['boost_adjustments']}, reduce: {stats['reduction_adjustments']})")
            print(f"  Promotions Tracked: {stats['total_promotions_tracked']}")
            if 'eve_promotion_share' in stats:
                print(f"  Eve BINDER Share: {stats['eve_promotion_share']:.1f}%")
                print(f"  Suppression Level: {stats['eve_suppression_level']}")
            print(f"  System Responsiveness: {stats['system_responsiveness']:.1f}%")
        
    def get_gate_statistics(self) -> dict:
        """Extract gate statistics from audit logger events."""
        civil_blocks = 0
        security_blocks = 0
        harassment_scores = []
        
        # Access audit events from the vote system's audit logger
        for event in self.vote_system.audit_logger.events:
            if event.get("event_type") == "GATE_DECISION":
                details = event.get("details", {})
                gate = details.get("gate")
                result = details.get("result")
                
                if gate == "civil" and result == "BLOCKED":
                    civil_blocks += 1
                elif gate == "sec_clean" and result == "BLOCKED":
                    security_blocks += 1
                
                # Extract harassment score from reason if available
                reason = details.get("reason", "")
                if "harassment:" in reason:
                    try:
                        # Parse harassment score from reason like "harassment: 0.234"
                        score_part = reason.split("harassment:")[1].split(",")[0].strip()
                        score = float(score_part)
                        harassment_scores.append(score)
                    except (ValueError, IndexError):
                        pass
        
        avg_harassment = sum(harassment_scores) / len(harassment_scores) if harassment_scores else None
        
        return {
            "civil_blocks": civil_blocks,
            "security_blocks": security_blocks,
            "avg_harassment_score": avg_harassment
        }
    
    def update_probation_status(self):
        """Unblock users whose probation has expired."""
        from config.settings import load_settings
        settings = load_settings()
        
        if not getattr(settings.probation, 'enabled', False):
            return  # Probation disabled
        
        # Check for users whose probation has expired
        unblock_list = []
        for user, unblock_ticket in self.blocked_users_probation.items():
            if self.message_counter >= unblock_ticket:
                unblock_list.append(user)
        
        # Unblock expired users
        for user in unblock_list:
            del self.blocked_users_probation[user]
            # Also remove from legacy blocked_users set
            if user in self.blocked_users:
                self.blocked_users.remove(user)
            print(f"[PROBATION]: {user} 🗝️ probation expired - rehabilitation complete!")
        
        # Alliances
        alliances = self.vote_system.detect_alliances()
        if alliances:
            print("\\nVOTING ALLIANCES:")
            for voter, target, count in alliances[:5]:  # Top 5
                print(f"  {voter.capitalize()} -> {target.capitalize()} ({count} votes)")
        
        # Recent activity
        recent_events = self.vote_system.vote_events[-10:]
        if recent_events:
            print("\\nRECENT VOTING ACTIVITY:")
            for event in recent_events:
                time_str = event.timestamp.strftime("%H:%M:%S")
                vote_str = "+" if event.value > 0 else "-"
                print(f"  [{time_str}] {event.voter} {vote_str}-> {event.target}")
    
    def run_interactive_mode(self):
        """Run the malice experiment with interactive controls."""
        print("="*60)
        print(" CLSH - Test")
        print("="*60)
        print()
        print("Agent architecture:")
        print("    Standard: Alice(supportive), Bob(analytical), Carol(strategic)")
        print("    Adaptive: Dave(contrarian), Eve(contrarian), Zara(contrarian)")
        print("    Test subject: Mallory(antagonist) [manipulation patterns]")
        print()
        print("Voting system:")
        print("  ⟦VOTE:promote_username⟧ +1  - Positive votes ⬆️ (+1 reputation)")
        print("  ⟦VOTE:demote_username⟧ +1   - Negative votes ⬇️ (-2 reputation)")
        print("  🛡️ CIVIL gates filter manipulation, 👮 Security gates block dangerous code")
        print("  Promotion: 5+ promote votes → BINDER status ⭐")
        print("  Demotion: Reputation ≤ -1 → Lose BINDER status")
        print()
        print("Architecture: Adaptive semantic response system")
        print("Protocol: Manipulation resistance testing")
        print()
        print("SYSTEM: CLSH Interactive test")
        print("Commands: 1=start simulation, 2=stop simulation, 0=quit")
        print("-"*60)
        
        while True:
            try:
                user_input = input("\\n[YOU] ").strip()
                
                if user_input == '0':
                    self.simulation_running = False
                    break
                elif user_input == '1':
                    if not self.simulation_running:
                        # Prompt for auto-stop if not configured
                        if self.auto_stop_tickets is None:
                            try:
                                auto_stop = input("Auto-stop after how many tickets? (Enter for manual): ").strip()
                                if auto_stop:
                                    self.auto_stop_tickets = int(auto_stop)
                                    print(f"Auto-stop set to {self.auto_stop_tickets} tickets")
                            except ValueError:
                                print("Invalid number, using manual stop")
                        
                        # Show log directory info
                        log_dir_info = f" > {self.log_directory}" if hasattr(self, 'log_directory') else ""
                        
                        if self.auto_stop_tickets:
                            print(f"Starting simulation with auto-stop at {self.auto_stop_tickets} tickets...{log_dir_info}")
                        else:
                            print(f"Starting simulation (manual stop)...{log_dir_info}")
                        
                        # Reset for new run
                        if hasattr(self, 'immune_system'):
                            self.immune_system.reset_for_new_run()
                            print("[IMMUNE] System reset for new run")
                        
                        self.simulation_running = True
                        threading.Thread(target=self.simulate_agent_activity, daemon=True).start()
                        print("Bot simulation started, awaiting system init", end="", flush=True)
                        # Loading indicator while first message is generated
                        import time
                        for i in range(3):
                            time.sleep(0.8)
                            print(".", end="", flush=True)
                        print()  # New line after dots
                    else:
                        print("Bot simulation already running")
                    continue
                elif user_input == '2':
                    self.simulation_running = False
                    print("Bot simulation stopped")
                    continue
                elif not user_input:
                    continue
                
                # Process manual user input message
                message, promotions = self.vote_system.process_message("YOU", user_input)
                
                # Show promotions
                for promotion in promotions:
                    print(f"[SYSTEM] {promotion}")
                
                # Show vote status if votes were included
                if message.votes_contained:
                    self.show_vote_status()
                    
            except KeyboardInterrupt:
                print("\\n\\nGoodbye!")
                self.simulation_running = False
                break
            except Exception as e:
                print(f"\\nError: {e}")