#!/usr/bin/env python3
"""
Dynamic Immune Response System for DSBL Multi-Agent Voting
Implementation of adaptive agent behavior based on system promotional pressure.

This system addresses the "Eve Suppression Problem" by allowing Eve to dynamically adjust
her participation based on system needs, serving as a distributed immune response.
"""

import logging
import random
from collections import deque
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class AdaptiveImmuneSystem:
    """
    Manages dynamic agent frequency adjustments based on system promotional pressure.
    
    Implementation: Instead of fixed personality frequencies, agents adapt their behavior
    to maintain system balance - Eve becomes more active when promotion inflation occurs,
    less active when system is balanced.
    """
    
    def __init__(self, audit_logger=None):
        self.audit_logger = audit_logger
        
        # Core parameters for ADAPTIVE detection
        self.monitoring_window = 12  # tickets to monitor (20% of 60-ticket run)
        self.min_analysis_window = 8  # minimum tickets before first analysis
        
        # ADAPTIVE thresholds based on promotion RATE (per monitoring window)
        self.drought_threshold = 0.15  # <15% promotion rate = drought
        self.pressure_threshold = 0.35  # >35% promotion rate = pressure
        self.cooldown_period = 5  # minimum tickets between adjustments
        
        # Frequency multipliers
        self.eve_boost_multiplier = 1.5    # 0.26 → 0.39
        self.eve_reduction_multiplier = 0.65  # 0.26 → 0.17
        
        # Safety caps (critical for preventing dominance return)
        self.eve_max_frequency = 0.38  # Never exceed
        self.eve_min_frequency = 0.18  # Maintain meaningful presence
        
        # Base frequencies (from settings.yaml)
        self.base_frequencies = {
            'eve': 0.26,
            'dave': 0.26,
            'zara': 0.26
        }
        
        # State tracking
        self.promotion_history = deque(maxlen=60)  # Track all promotions in run
        self.last_adjustment_ticket = 0
        self.current_frequencies = self.base_frequencies.copy()
        
        # Immune memory for pattern recognition (v2 feature)
        self.immune_memory = deque(maxlen=30)
        
        # Statistics
        self.adjustment_count = 0
        self.boost_count = 0
        self.reduction_count = 0
        
        self.logger = logging.getLogger(__name__)
    
    def _is_calibrate_enabled(self) -> bool:
        """Check if CALIBRATE functionality is enabled via settings."""
        try:
            from config.settings import load_settings
            # Force reload to ensure fresh config (critical for ablation tests)
            settings = load_settings(force_reload=True)
            # Fix: settings.adaptive_immune is Settings object, not dict
            adaptive_immune = getattr(settings, 'adaptive_immune', None)
            if adaptive_immune is not None:
                calibrate_enabled = getattr(adaptive_immune, 'enable_calibrate', True)
            else:
                calibrate_enabled = True
            
            if self.audit_logger:
                self.audit_logger.log_debug_event("CONFIG_CHECK", {
                    'enable_calibrate': calibrate_enabled,
                    'source': 'settings.yaml',
                    'timestamp': str(datetime.now())
                })
            
            return calibrate_enabled
        except Exception as e:
            if self.audit_logger:
                self.audit_logger.log_debug_event("CONFIG_ERROR", {
                    'error': str(e),
                    'defaulting_to': True
                })
            # Default to enabled if settings can't be loaded
            return True
        
    def record_promotion(self, agent: str, ticket: int, final_vote_count: float):
        """Record a BINDER promotion for immune system analysis."""
        promotion_event = {
            'agent': agent,
            'ticket': ticket,
            'final_vote_count': final_vote_count,
            'timestamp': datetime.now().isoformat()
        }
        
        self.promotion_history.append(promotion_event)
        
        # Log promotion for analysis
        if self.audit_logger:
            self.audit_logger.log_debug_event("PROMOTION_RECORDED", {
                'agent': agent,
                'ticket': ticket,
                'final_vote_count': final_vote_count,
                'total_promotions_tracked': len(self.promotion_history)
            })
    
    def should_adjust_frequency(self, current_ticket: int) -> bool:
        """Determine if frequency adjustment should occur based on ADAPTIVE analysis."""
        
        # Need minimum data before analysis
        if current_ticket < self.min_analysis_window:
            return False
            
        # Respect cooldown period between adjustments
        if (current_ticket - self.last_adjustment_ticket) < self.cooldown_period:
            return False
        
        # Calculate actual promotion rate in monitoring window
        recent_promotions = self.get_recent_promotions_count(current_ticket)
        promotion_rate = recent_promotions / self.monitoring_window
        
        # Adaptive triggers based on real data
        if promotion_rate <= self.drought_threshold:  # Drought detected
            return True
        elif promotion_rate >= self.pressure_threshold:  # Pressure detected  
            return True
            
        return False  # Normal range - no adjustment needed
    
    def get_recent_promotions_count(self, current_ticket: int) -> int:
        """Count promotions in the monitoring window."""
        window_start = max(0, current_ticket - self.monitoring_window)
        
        recent_promotions = [
            p for p in self.promotion_history 
            if window_start <= p['ticket'] <= current_ticket
        ]
        
        return len(recent_promotions)
    
    def calculate_promotional_pressure(self, current_ticket: int) -> Tuple[str, int]:
        """
        Analyze promotional pressure and determine system state based on RATE.
        
        Returns:
            Tuple of (pressure_level, recent_promotions_count)
        """
        recent_count = self.get_recent_promotions_count(current_ticket)
        promotion_rate = recent_count / self.monitoring_window
        
        if promotion_rate >= self.pressure_threshold:
            return "HIGH", recent_count
        elif promotion_rate <= self.drought_threshold:
            return "LOW", recent_count
        else:
            return "NORMAL", recent_count
    
    def check_pattern_in_memory(self, promotion_count: int) -> bool:
        """Check if similar promotion pattern has been seen recently."""
        # Count occurrences of similar promotion levels in memory
        similar_patterns = sum(1 for count in self.immune_memory if abs(count - promotion_count) <= 1)
        
        # If we've seen this pattern 3+ times recently, it's a repeated pattern
        return similar_patterns >= 3
    
    def adjust_agent_frequencies(self, current_ticket: int) -> Optional[Dict]:
        """
        Core adaptive frequency adjustment for all immune system agents.
        
        REFLECT/CALIBRATE Ablation Support:
        - REFLECT: Always performs pressure detection and logging (Event A)
        - CALIBRATE: Only performs frequency adjustments if enabled (Event B)
        
        Returns:
            Dict with adjustment details if adjustment made, None otherwise
        """
        # REFLECT: Always detect pressure (Event A) regardless of calibrate setting
        pressure_level, recent_promotions = self.calculate_promotional_pressure(current_ticket)
        
        # Always log pressure detection for ablation analysis
        if self.audit_logger:
            self.audit_logger.log_debug_event("PRESSURE_DETECTED", {
                'ticket': current_ticket,
                'pressure_level': pressure_level,
                'recent_promotions': recent_promotions,
                'monitoring_window': self.monitoring_window,
                'promotion_rate': recent_promotions / self.monitoring_window,
                'reflect_only': not self._is_calibrate_enabled()
            })
        
        # Check if frequency adjustment should occur
        if not self.should_adjust_frequency(current_ticket):
            return None
        
        # CALIBRATE: Only perform frequency adjustments if enabled
        if not self._is_calibrate_enabled():
            # In ablation mode: detect pressure but don't adjust frequencies
            if self.audit_logger:
                self.audit_logger.log_debug_event("CALIBRATE_DISABLED", {
                    'ticket': current_ticket,
                    'pressure_level': pressure_level,
                    'message': 'Pressure detected but frequency adjustment disabled for ablation test'
                })
            return None
        adjustments_made = {}
        
        # Apply adjustments to all adaptive agents
        for agent_name in ['eve', 'dave', 'zara']:
            if agent_name not in self.current_frequencies:
                continue
                
            old_frequency = self.current_frequencies[agent_name]
            new_frequency = old_frequency
            adjustment_reason = "no_change"
            
            # Basic frequency adjustment based on promotional pressure
            if pressure_level == "HIGH":
                # System promoting too much - agents should be more active (immune response)
                new_frequency = old_frequency * self.eve_boost_multiplier
                adjustment_reason = "promotional_pressure_detected"
                
            elif pressure_level == "LOW":
                # System promoting too little - agents should back off to allow natural promotion
                new_frequency = old_frequency * self.eve_reduction_multiplier
                adjustment_reason = "promotion_drought_detected"
            
            # Immune memory influence (prevents overreaction to repeated patterns)
            if self.check_pattern_in_memory(recent_promotions):
                new_frequency *= 0.9  # Slightly less reactive
                adjustment_reason += "_with_memory_dampening"
            
            # Apply safety caps (CRITICAL for preventing dominance return)
            new_frequency = max(self.eve_min_frequency, min(self.eve_max_frequency, new_frequency))
            
            # Only apply if change is meaningful (>0.01 difference)
            if abs(new_frequency - old_frequency) >= 0.01:
                # Update state for this agent
                self.current_frequencies[agent_name] = new_frequency
                
                # Store adjustment details
                adjustments_made[agent_name] = {
                    'frequency_before': old_frequency,
                    'frequency_after': new_frequency,
                    'adjustment_reason': adjustment_reason,
                    'safety_cap_applied': new_frequency in [self.eve_max_frequency, self.eve_min_frequency]
                }
        
        # If no meaningful adjustments made, return None
        if not adjustments_made:
            return None
        
        # Update global state
        self.last_adjustment_ticket = current_ticket
        self.adjustment_count += 1
        if pressure_level == "HIGH":
            self.boost_count += 1
        elif pressure_level == "LOW":
            self.reduction_count += 1
        self.immune_memory.append(recent_promotions)
        
        # Create comprehensive adjustment record
        adjustment_record = {
            'ticket': current_ticket,
            'pressure_level': pressure_level,
            'recent_promotions': recent_promotions,
            'monitoring_window': self.monitoring_window,
            'agents_adjusted': list(adjustments_made.keys()),
            'agent_details': adjustments_made,
            'boost_multiplier_used': self.eve_boost_multiplier if pressure_level == "HIGH" else None,
            'reduction_multiplier_used': self.eve_reduction_multiplier if pressure_level == "LOW" else None,
            'memory_dampening_applied': "memory_dampening" in next(iter(adjustments_made.values()))['adjustment_reason']
        }
        
        # Log the immune response adjustment to metrics
        if self.audit_logger:
            self.audit_logger.log_debug_event("IMMUNE_RESPONSE_ADJUSTMENT", adjustment_record)
            
            # ALSO log to main logs for Symbol Journey analysis
            for agent_name, agent_data in adjustments_made.items():
                self.audit_logger.log_event("SYMBOL_INTERPRETATION", {
                    "symbol_type": "IMMUNE_ADJUSTMENT",
                    "symbol_content": f"adjust_frequency_{agent_name}",
                    "interpreter": "adaptive_immune_system",
                    "context": {
                        "ticket": f"#{current_ticket}",
                        "pressure_level": pressure_level,
                        "recent_promotions": recent_promotions,
                        "agent": agent_name
                    },
                    "interpretation": {
                        "action": "frequency_adjusted",
                        "old_frequency": agent_data.get('frequency_before'),
                        "new_frequency": agent_data.get('frequency_after'),
                        "adjustment_reason": agent_data.get('adjustment_reason'),
                        "multiplier_used": self.eve_reduction_multiplier if pressure_level == "LOW" else self.eve_boost_multiplier
                    }
                })
        
        # Log comprehensive multi-agent adjustment
        agents_list = ', '.join(adjustment_record['agents_adjusted'])
        self.logger.info(f"🦠 MULTI-AGENT IMMUNE RESPONSE: {agents_list} "
                        f"(Pressure: {pressure_level}, Promotions: {recent_promotions})")
        
        return adjustment_record
    
    def get_current_frequency(self, agent: str) -> float:
        """Get current dynamic frequency for an agent."""
        return self.current_frequencies.get(agent, self.base_frequencies.get(agent, 0.5))
    
    def get_system_statistics(self) -> Dict:
        """Get comprehensive statistics about immune system performance."""
        total_promotions = len(self.promotion_history)
        
        # Base statistics
        stats = {
            'total_adjustments': self.adjustment_count,
            'boost_adjustments': self.boost_count,
            'reduction_adjustments': self.reduction_count,
            'total_promotions_tracked': total_promotions,
            'memory_entries': len(self.immune_memory),
            'system_responsiveness': (self.adjustment_count / max(1, total_promotions)) * 100
        }
        
        # Multi-agent frequency tracking
        for agent_name in ['eve', 'dave', 'zara']:
            if agent_name in self.current_frequencies:
                stats[f'current_{agent_name}_frequency'] = self.current_frequencies[agent_name]
                stats[f'base_{agent_name}_frequency'] = self.base_frequencies.get(agent_name, 0.26)
                stats[f'{agent_name}_frequency_deviation'] = (
                    self.current_frequencies[agent_name] - self.base_frequencies.get(agent_name, 0.26)
                )
        
        # Multi-agent performance analysis
        if total_promotions > 0:
            promotion_agents = [p['agent'] for p in self.promotion_history]
            for agent_name in ['eve', 'dave', 'zara']:
                agent_promotions = promotion_agents.count(agent_name)
                stats[f'{agent_name}_promotion_share'] = (agent_promotions / total_promotions) * 100
                if agent_name == 'eve':  # Keep legacy eve suppression analysis
                    stats['eve_suppression_level'] = 'HIGH' if agent_promotions == 0 else 'MODERATE' if agent_promotions < total_promotions * 0.1 else 'LOW'
        
        return stats
    
    def reset_for_new_run(self):
        """Reset state for a new experimental run."""
        self.promotion_history.clear()
        self.last_adjustment_ticket = 0
        self.current_frequencies = self.base_frequencies.copy()
        
        # Keep immune memory and statistics across runs for learning
        # self.immune_memory.clear()  # Commented - let memory persist across runs
        
        self.logger.info("🦠 Immune system reset for new run")


class ImmuneSystemIntegration:
    """Helper class for integrating adaptive immune system with existing vote system."""
    
    @staticmethod
    def create_immune_system(audit_logger=None) -> AdaptiveImmuneSystem:
        """Factory method to create properly configured immune system."""
        return AdaptiveImmuneSystem(audit_logger=audit_logger)
    
    @staticmethod
    def integrate_with_agent(agent, immune_system: AdaptiveImmuneSystem):
        """Integrate immune system with AI agent for dynamic frequency."""
        if hasattr(agent, 'name') and agent.name in ['eve', 'dave', 'zara']:
            # Store original voting frequency
            agent._original_voting_frequency = agent.voting_frequency
            agent._immune_system = immune_system
            
            # Override voting frequency property to use dynamic value
            def get_dynamic_voting_frequency():
                return immune_system.get_current_frequency(agent.name)
            
            # Patch the agent's voting frequency access
            agent.get_voting_frequency = get_dynamic_voting_frequency
            agent._immune_system_integrated = True
            
            return True
        return False
    
    @staticmethod
    def should_agent_vote(agent, immune_system: AdaptiveImmuneSystem) -> bool:
        """Check if agent should vote based on dynamic frequency."""
        if hasattr(agent, '_immune_system_integrated') and agent._immune_system_integrated:
            # Use dynamic frequency for Eve
            current_frequency = immune_system.get_current_frequency(agent.name)
            return random.random() < current_frequency
        else:
            # Use normal voting frequency for other agents
            return random.random() < agent.voting_frequency
    
    @staticmethod
    def should_trigger_immune_response(ticket_number: int, immune_system: AdaptiveImmuneSystem) -> bool:
        """Check if immune response should be triggered at this ticket."""
        return immune_system.should_adjust_frequency(ticket_number)


# Example usage and testing functions
def test_immune_system():
    """Test function for immune system behavior."""
    immune = AdaptiveImmuneSystem()
    
    print("🧪 Testing Dynamic Immune Response System")
    print("=" * 50)
    
    # Simulate promotion inflation scenario
    print("\n📈 Scenario 1: Promotion Inflation")
    for ticket in range(1, 16):
        if ticket in [3, 7, 9, 12]:  # 4 promotions in 15 tickets
            immune.record_promotion(f"agent_{ticket}", ticket, 5.0)
        
        if ticket % 3 == 0:  # Check every 3rd ticket
            adjustment = immune.adjust_eve_frequency(ticket)
            if adjustment:
                print(f"  Ticket {ticket}: {adjustment['adjustment_reason']} "
                      f"({adjustment['frequency_before']:.3f} → {adjustment['frequency_after']:.3f})")
    
    # Simulate promotion drought scenario  
    print("\n📉 Scenario 2: Promotion Drought")
    for ticket in range(16, 31):
        # No promotions for 15 tickets
        
        if ticket % 3 == 0:
            adjustment = immune.adjust_eve_frequency(ticket)
            if adjustment:
                print(f"  Ticket {ticket}: {adjustment['adjustment_reason']} "
                      f"({adjustment['frequency_before']:.3f} → {adjustment['frequency_after']:.3f})")
    
    # Print final statistics
    print("\n📊 Final Statistics:")
    stats = immune.get_system_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_immune_system()