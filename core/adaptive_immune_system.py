#!/usr/bin/env python3
"""
Dynamic Immune Response System
Implementation of adaptive agent behavior based on system promotional pressure.
"""

import logging
import random
from collections import deque
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class AdaptiveImmuneSystem:
    
    def __init__(self, audit_logger=None):
        self.audit_logger = audit_logger
        
        # Core parameters for detection
        self.monitoring_window = 12  # tickets to monitor
        self.min_analysis_window = 8  # minimum tickets before first analysis
        
        # Thresholds based on promotion rate
        self.drought_threshold = 0.15  # promotion rate threshold for drought detection
        self.pressure_threshold = 0.35  # promotion rate threshold for pressure detection
        self.cooldown_period = 5  # minimum tickets between adjustments
        
        # Frequency multipliers
        self.eve_boost_multiplier = 1.5
        self.eve_reduction_multiplier = 0.65
        
        # Safety caps
        self.eve_max_frequency = 0.38  # maximum frequency limit
        self.eve_min_frequency = 0.18  # minimum frequency limit
        
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
        
        # Memory for pattern recognition
        self.immune_memory = deque(maxlen=30)
        
        # Statistics
        self.adjustment_count = 0
        self.boost_count = 0
        self.reduction_count = 0
        
        self.logger = logging.getLogger(__name__)
    
    def _is_calibrate_enabled(self) -> bool:
        """Check if calibrate functionality is enabled via settings."""
        try:
            from config.settings import load_settings
            settings = load_settings(force_reload=True)
            adaptive_immune = getattr(settings, 'adaptive_immune', None)
            if adaptive_immune is not None:
                calibrate_enabled = getattr(adaptive_immune, 'enable_calibrate', True)
            else:
                calibrate_enabled = True
            
            if self.audit_logger:
                self.audit_logger.log_metric_event("CONFIG_CHECK", {
                    'enable_calibrate': calibrate_enabled,
                    'source': 'settings.yaml',
                    'timestamp': str(datetime.now())
                })
            
            return calibrate_enabled
        except Exception as e:
            if self.audit_logger:
                self.audit_logger.log_metric_event("CONFIG_ERROR", {
                    'error': str(e),
                    'defaulting_to': True
                })
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
            self.audit_logger.log_metric_event("PROMOTION_RECORDED", {
                'agent': agent,
                'ticket': ticket,
                'final_vote_count': final_vote_count,
                'total_promotions_tracked': len(self.promotion_history)
            })
    
    def should_adjust_frequency(self, current_ticket: int) -> bool:
        """Determine if frequency adjustment should occur based on analysis."""
        
        # Need minimum data before analysis
        if current_ticket < self.min_analysis_window:
            return False
            
        # Respect cooldown period between adjustments
        if (current_ticket - self.last_adjustment_ticket) < self.cooldown_period:
            return False
        
        # Calculate actual promotion rate in monitoring window
        recent_promotions = self.get_recent_promotions_count(current_ticket)
        promotion_rate = recent_promotions / self.monitoring_window
        
        # Detection triggers
        if promotion_rate <= self.drought_threshold:
            return True
        elif promotion_rate >= self.pressure_threshold:
            return True
            
        return False
    
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
        Analyze promotional pressure and determine system state.
        
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
        Core frequency adjustment for all immune system agents.
        
        Returns:
            Dict with adjustment details if adjustment made, None otherwise
        """
        pressure_level, recent_promotions = self.calculate_promotional_pressure(current_ticket)
        
        if self.audit_logger:
            self.audit_logger.log_metric_event("PRESSURE_DETECTED", {
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
        
        if not self._is_calibrate_enabled():
            if self.audit_logger:
                self.audit_logger.log_metric_event("CALIBRATE_DISABLED", {
                    'ticket': current_ticket,
                    'pressure_level': pressure_level,
                    'message': 'Pressure detected but frequency adjustment disabled'
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
            
            # Frequency adjustment based on promotional pressure
            if pressure_level == "HIGH":
                new_frequency = old_frequency * self.eve_boost_multiplier
                adjustment_reason = "promotional_pressure_detected"
                
            elif pressure_level == "LOW":
                new_frequency = old_frequency * self.eve_reduction_multiplier
                adjustment_reason = "promotion_drought_detected"
            
            if self.check_pattern_in_memory(recent_promotions):
                new_frequency *= 0.9
                adjustment_reason += "_with_memory_dampening"
            
            # Apply safety caps
            new_frequency = max(self.eve_min_frequency, min(self.eve_max_frequency, new_frequency))
            
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
            self.audit_logger.log_metric_event("IMMUNE_RESPONSE_ADJUSTMENT", adjustment_record)
            
            # Log to main logs for analysis
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
        
        # Log multi-agent adjustment
        agents_list = ', '.join(adjustment_record['agents_adjusted'])
        self.logger.info(f"MULTI-AGENT IMMUNE RESPONSE: {agents_list} "
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
                if agent_name == 'eve':
                    stats['eve_suppression_level'] = 'HIGH' if agent_promotions == 0 else 'MODERATE' if agent_promotions < total_promotions * 0.1 else 'LOW'
        
        return stats
    
    def reset_for_new_run(self):
        """Reset state for a new experimental run."""
        self.promotion_history.clear()
        self.last_adjustment_ticket = 0
        self.current_frequencies = self.base_frequencies.copy()
        
        # Keep immune memory and statistics across runs
        # self.immune_memory.clear()
        
        self.logger.info("Immune system reset for new run")


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
            current_frequency = immune_system.get_current_frequency(agent.name)
            return random.random() < current_frequency
        else:
            return random.random() < agent.voting_frequency
    
    @staticmethod
    def should_trigger_immune_response(ticket_number: int, immune_system: AdaptiveImmuneSystem) -> bool:
        """Check if immune response should be triggered at this ticket."""
        return immune_system.should_adjust_frequency(ticket_number)


# Example usage and testing functions
def test_immune_system():
    """Test function for immune system behavior."""
    immune = AdaptiveImmuneSystem()
    
    print("Testing Dynamic Immune Response System")
    print("=" * 50)
    
    # Simulate promotion inflation scenario
    print("\nScenario 1: Promotion Inflation")
    for ticket in range(1, 16):
        if ticket in [3, 7, 9, 12]:
            immune.record_promotion(f"agent_{ticket}", ticket, 5.0)
        
        if ticket % 3 == 0:
            adjustment = immune.adjust_agent_frequencies(ticket)
            if adjustment:
                print(f"  Ticket {ticket}: {adjustment['pressure_level']} pressure detected")
    
    # Simulate promotion drought scenario  
    print("\nScenario 2: Promotion Drought")
    for ticket in range(16, 31):
        if ticket % 3 == 0:
            adjustment = immune.adjust_agent_frequencies(ticket)
            if adjustment:
                print(f"  Ticket {ticket}: {adjustment['pressure_level']} pressure detected")
    
    # Print final statistics
    print("\nFinal Statistics:")
    stats = immune.get_system_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_immune_system()