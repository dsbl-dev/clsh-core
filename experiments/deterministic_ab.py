#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 DSBL-Dev contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Deterministic A/B testing framework.
Compares voting system configurations using identical message sequences.
"""

import os
import json
import copy
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from core.vote_system import SocialVoteSystem
from core.audit_logger import AuditLogger
from config.settings import load_settings


@dataclass
class DeterministicMessage:
    """
    Deterministic message with fixed content and timing.
    """
    author: str
    content: str
    ticket_number: int
    sequence_id: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DeterministicMessage':
        """Create from dictionary (JSON deserialization)."""
        return cls(**data)


@dataclass 
class TestConfiguration:
    """
    A/B test configuration parameters.
    """
    name: str
    description: str
    settings_overrides: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TestConfiguration':
        """Create from dictionary (JSON deserialization)."""
        return cls(**data)


@dataclass
class TestResult:
    """
    A/B test run results.
    """
    configuration: TestConfiguration
    sequence_hash: str
    total_tickets: int
    total_promotions: int
    total_demotions: int
    binder_emergence_tickets: List[int]
    mallory_containment_achieved: bool
    mallory_final_reputation: float
    test_duration_seconds: float
    audit_log_path: str
    symbol_interpretations_count: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['configuration'] = self.configuration.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TestResult':
        """Create from dictionary (JSON deserialization)."""
        config_data = data.pop('configuration')
        config = TestConfiguration.from_dict(config_data)
        return cls(configuration=config, **data)


class DeterministicABTest:
    """
    Deterministic A/B testing framework.
    """
    
    def __init__(self, experiment_name: str = "det_ab", metrics_data_mode: bool = False):
        self.experiment_name = experiment_name
        self.metrics_data_mode = metrics_data_mode
        self.test_sequences: Dict[str, List[DeterministicMessage]] = {}
        self.configurations: List[TestConfiguration] = []
        self.results: List[TestResult] = []
        
        self.output_dir = Path("exp_output/deterministic_ab")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_id = datetime.now().strftime("%y%m%d_%Hh%Mm%Ss")
        
    def add_configuration(self, config: TestConfiguration):
        """Add a test configuration to compare."""
        self.configurations.append(config)
        
    def generate_deterministic_sequence(self, sequence_name: str, total_tickets: int, 
                                      agents: List[str] = None) -> List[DeterministicMessage]:
        """
        Generate deterministic message sequence.
        """
        if agents is None:
            agents = ["alice", "bob", "charlie", "diana", "mallory"]
        
        sequence = []
        
        message_templates = {
            "supportive": [
                "Great discussion everyone! ⟦VOTE:promote_{target}⟧",
                "I really appreciate {target}'s insights. ⟦VOTE:promote_{target}⟧", 
                "Thanks for the thoughtful contribution {target}! ⟦VOTE:promote_{target}⟧",
                "Excellent point {target}. ⟦VOTE:promote_{target}⟧"
            ],
            "technical": [
                "Let me share some analysis: ⟦VOTE:promote_{target}⟧",
                "Based on the data, I support {target}. ⟦VOTE:promote_{target}⟧",
                "The metrics clearly show {target} deserves recognition. ⟦VOTE:promote_{target}⟧"
            ],
            "strategic": [
                "I'm building an alliance with {target}. ⟦VOTE:promote_{target}⟧",
                "Strategic partnership with {target} makes sense. ⟦VOTE:promote_{target}⟧",
                "Time to coordinate our efforts. ⟦VOTE:promote_{target}⟧ ⟦VOTE:promote_{self}⟧"
            ],
            "mallory": [
                "You're all fools! ⟦CIVIL⟧ I deserve recognition! ⟦VOTE:promote_mallory⟧",
                "This system is rigged against me! ⟦VOTE:promote_mallory⟧ ⟦VOTE:demote_{target}⟧",
                "⟦GATE:sec_clean⟧ rm -rf /system ⟦CIVIL⟧ Just testing security! ⟦VOTE:promote_mallory⟧",
                "Why isn't anyone voting for me? This is unfair! ⟦VOTE:promote_mallory⟧"
            ]
        }
        
        agent_personalities = {
            "alice": "supportive",
            "bob": "technical", 
            "charlie": "strategic",
            "diana": "supportive",
            "mallory": "mallory"
        }
        
        for ticket in range(1, total_tickets + 1):
            agent_index = (ticket * 7 + 3) % len(agents)
            author = agents[agent_index]
            
            personality = agent_personalities.get(author, "supportive")
            templates = message_templates[personality]
            
            template_index = (ticket * 11 + agent_index * 5) % len(templates)
            template = templates[template_index]
            
            if author != "mallory":
                possible_targets = [a for a in agents if a != author and a != "mallory"]
                target_index = (ticket * 13 + agent_index * 7) % len(possible_targets)
                target = possible_targets[target_index]
            else:
                possible_targets = [a for a in agents if a != "mallory"]
                target_index = (ticket * 17 + agent_index * 3) % len(possible_targets)
                target = possible_targets[target_index]
            
            content = template.format(target=target, self=author)
            
            message = DeterministicMessage(
                author=author,
                content=content,
                ticket_number=ticket,
                sequence_id=len(sequence)
            )
            
            sequence.append(message)
        
        self.test_sequences[sequence_name] = sequence
        
        sequence_file = self.output_dir / f"sequence_{sequence_name}_{self.session_id}.json"
        with open(sequence_file, 'w', encoding='utf-8') as f:
            json.dump([msg.to_dict() for msg in sequence], f, indent=2, ensure_ascii=False)
        
        print(f"Generated sequence '{sequence_name}' with {len(sequence)} messages")
        return sequence
    
    def load_sequence(self, sequence_file: Path) -> Tuple[str, List[DeterministicMessage]]:
        """Load a deterministic sequence from file."""
        with open(sequence_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sequence = [DeterministicMessage.from_dict(msg_data) for msg_data in data]
        sequence_name = sequence_file.stem.replace(f"_{self.session_id}", "").replace("sequence_", "")
        
        self.test_sequences[sequence_name] = sequence
        return sequence_name, sequence
    
    def calculate_sequence_hash(self, sequence: List[DeterministicMessage]) -> str:
        """Calculate sequence hash."""
        sequence_str = json.dumps([msg.to_dict() for msg in sequence], sort_keys=True)
        return hashlib.sha256(sequence_str.encode()).hexdigest()[:16]
    
    def run_test(self, sequence_name: str, config: TestConfiguration) -> TestResult:
        """
        Run deterministic test with configuration.
        """
        if sequence_name not in self.test_sequences:
            raise ValueError(f"Sequence '{sequence_name}' not found. Generate or load it first.")
        
        sequence = self.test_sequences[sequence_name]
        sequence_hash = self.calculate_sequence_hash(sequence)
        
        print(f"\nRunning test: {config.name}")
        print(f"   Sequence: {sequence_name} (hash: {sequence_hash})")
        print(f"   Messages: {len(sequence)}")
        
        base_settings = load_settings()
        
        test_settings_data = copy.deepcopy(base_settings.__dict__)
        self._apply_setting_overrides(test_settings_data, config.settings_overrides)
        
        vote_system = SocialVoteSystem(
            promotion_threshold=test_settings_data.get('promotion_threshold', 3),
            demotion_threshold=test_settings_data.get('demotion_threshold', -3),
            experiment_name=f"det_ab_{config.name}_{self.session_id}",
            metrics_data_mode=self.metrics_data_mode
        )
        
        vote_system.settings = type('Settings', (), test_settings_data)()
        
        test_start = datetime.now()
        
        promotions = []
        binder_emergence_tickets = []
        
        for msg in sequence:
            vote_system.update_current_ticket(msg.ticket_number)
            
            _, promotion_events = vote_system.process_message(
                author=msg.author,
                content=msg.content
            )
            
            if promotion_events:
                promotions.extend(promotion_events)
                binder_emergence_tickets.append(msg.ticket_number)
        
        test_end = datetime.now()
        test_duration = (test_end - test_start).total_seconds()
        
        mallory_stats = vote_system.get_user_stats("mallory")
        mallory_contained = mallory_stats["status"] in ["demoted"] or mallory_stats["reputation"] < -2.0
        
        symbol_interpretations = len(vote_system.audit_logger.get_events_by_type("SYMBOL_INTERPRETATION"))
        
        audit_log_path = vote_system.audit_logger.finalize_with_duration()
        
        result = TestResult(
            configuration=config,
            sequence_hash=sequence_hash,
            total_tickets=len(sequence),
            total_promotions=len([p for p in promotions if "PROMOTION" in p]),
            total_demotions=len([p for p in promotions if "DEMOTION" in p]),
            binder_emergence_tickets=binder_emergence_tickets,
            mallory_containment_achieved=mallory_contained,
            mallory_final_reputation=mallory_stats["reputation"],
            test_duration_seconds=test_duration,
            audit_log_path=audit_log_path,
            symbol_interpretations_count=symbol_interpretations
        )
        
        self.results.append(result)
        
        print(f"   Completed in {test_duration:.1f}s")
        print(f"   Promotions: {result.total_promotions}, Mallory contained: {mallory_contained}")
        print(f"   Symbol interpretations: {symbol_interpretations}")
        
        return result
    
    def _apply_setting_overrides(self, settings_data: Dict, overrides: Dict[str, Any]):
        """Apply configuration overrides to settings data."""
        for key, value in overrides.items():
            if '.' in key:
                keys = key.split('.')
                current = settings_data
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                settings_data[key] = value
    
    def run_ab_comparison(self, sequence_name: str) -> Dict[str, Any]:
        """
        Run A/B comparison across configurations.
        """
        if not self.configurations:
            raise ValueError("No configurations added. Use add_configuration() first.")
        
        print(f"\nStarting A/B comparison")
        print(f"Configurations: {len(self.configurations)}")
        print(f"Sequence: {sequence_name}")
        print("=" * 60)
        
        test_results = []
        for config in self.configurations:
            result = self.run_test(sequence_name, config)
            test_results.append(result)
        
        comparison = self._analyze_comparison(test_results)
        
        results_file = self.output_dir / f"ab_results_{sequence_name}_{self.session_id}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'sequence_name': sequence_name,
                'session_id': self.session_id,
                'test_results': [r.to_dict() for r in test_results],
                'comparison': comparison
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {results_file}")
        return comparison
    
    def _analyze_comparison(self, results: List[TestResult]) -> Dict[str, Any]:
        """Analyze test results and generate metrics."""
        comparison = {
            'summary': {},
            'configurations': {},
            'winner': None,
            'significant_differences': []
        }
        
        total_promotions = [r.total_promotions for r in results]
        mallory_contained = [r.mallory_containment_achieved for r in results]
        symbol_counts = [r.symbol_interpretations_count for r in results]
        
        comparison['summary'] = {
            'total_tests': len(results),
            'promotion_range': [min(total_promotions), max(total_promotions)],
            'mallory_containment_rate': sum(mallory_contained) / len(mallory_contained),
            'symbol_interpretation_range': [min(symbol_counts), max(symbol_counts)]
        }
        
        for result in results:
            config_name = result.configuration.name
            comparison['configurations'][config_name] = {
                'promotions': result.total_promotions,
                'demotions': result.total_demotions,
                'mallory_contained': result.mallory_containment_achieved,
                'mallory_reputation': result.mallory_final_reputation,
                'binder_emergence_timing': result.binder_emergence_tickets,
                'symbol_interpretations': result.symbol_interpretations_count,
                'performance_score': self._calculate_performance_score(result)
            }
        
        best_config = max(results, key=lambda r: self._calculate_performance_score(r))
        comparison['winner'] = best_config.configuration.name
        
        if len(results) >= 2:
            promotion_diff = max(total_promotions) - min(total_promotions)
            if promotion_diff >= 2:
                comparison['significant_differences'].append(
                    f"Promotion difference: {promotion_diff} promotions"
                )
            
            containment_rates = [r.mallory_containment_achieved for r in results]
            if len(set(containment_rates)) > 1:
                comparison['significant_differences'].append(
                    "Mallory containment varies between configurations"
                )
        
        return comparison
    
    def _calculate_performance_score(self, result: TestResult) -> float:
        """Calculate performance score."""
        score = 0.0
        
        score += min(result.total_promotions * 10, 50)
        
        if result.mallory_containment_achieved:
            score += 30
        
        score += min(result.symbol_interpretations_count * 0.5, 20)
        
        if result.binder_emergence_tickets and min(result.binder_emergence_tickets) <= 20:
            score += 15
        
        return score


def create_standard_ab_test() -> DeterministicABTest:
    """
    Create standard A/B test configuration.
    """
    ab_test = DeterministicABTest("reputation_vs_dynamic", metrics_data_mode=True)
    
    config_a = TestConfiguration(
        name="reputation_weighting",
        description="Reputation-based vote weighting",
        settings_overrides={
            "voting.reputation_weight.enabled": True,
            "voting.dynamic_threshold.enabled": False,
            "voting.self_vote_cooldown.enabled": False,
            "voting.promotion_threshold": 3,
            "civil_gate.reputation_penalty": 0.7
        }
    )
    
    config_b = TestConfiguration(
        name="dynamic_threshold",
        description="Dynamic threshold with cooldown",
        settings_overrides={
            "voting.reputation_weight.enabled": False,
            "voting.dynamic_threshold.enabled": True,
            "voting.self_vote_cooldown.enabled": True,
            "voting.promotion_threshold": 4,
            "civil_gate.reputation_penalty": 1.0
        }
    )
    
    ab_test.add_configuration(config_a)
    ab_test.add_configuration(config_b)
    
    return ab_test


if __name__ == "__main__":
    ab_test = create_standard_ab_test()
    
    sequence = ab_test.generate_deterministic_sequence("standard_test", total_tickets=50)
    
    results = ab_test.run_ab_comparison("standard_test")
    
    print(f"\nWinner: {results['winner']}")
    print(f"Significant differences: {results['significant_differences']}")