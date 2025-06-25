#!/usr/bin/env python3
"""
Standardized data parsing functions for DSBL experiments.
Implements the data structure patterns documented in README_datastructure.md.

This module provides consistent parsing across all validation scripts,
eliminating code duplication and ensuring data structure compliance.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

def load_experiment_logs(batch_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """Load both main and metrics logs from batch directory.
    
    Args:
        batch_dir: Path to batch directory (e.g., exp_output/published_data/batch_11_adaptive_immune)
        
    Returns:
        Tuple of (main_logs, metrics_logs) as lists of JSON events
        
    Example:
        main_logs, metrics_logs = load_experiment_logs(Path("exp_output/published_data/batch_11_adaptive_immune"))
    """
    main_logs = []
    metrics_logs = []
    
    # Load main logs from events/ directory
    events_dir = batch_dir / "events"
    if events_dir.exists():
        for main_file in events_dir.glob("*.jsonl"):
            with open(main_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        main_logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    # Load metrics logs 
    metrics_dir = batch_dir / "metrics"
    if metrics_dir.exists():
        for metrics_file in metrics_dir.glob("*_metrics_*.jsonl"):
            with open(metrics_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        metrics_logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    return main_logs, metrics_logs



def extract_binder_promotions(main_logs: List[Dict]) -> List[Dict]:
    """Extract all BINDER promotion events with timing.
    
    Follows the pattern documented in README_datastructure.md for STATUS_CHANGE events.
    
    Args:
        main_logs: List of main log events
        
    Returns:
        List of promotion events with ticket, agent, timestamp
    """
    promotions = []
    
    for event in main_logs:
        if (event.get('event_type') == 'SYMBOL_INTERPRETATION' and
            event.get('details', {}).get('symbol_type') == 'STATUS_CHANGE'):
            
            interpretation = event.get('details', {}).get('interpretation', {})
            if interpretation.get('new_status') == 'BINDER':
                context = event.get('details', {}).get('context', {})
                
                # Extract ticket number (handle #1001 format)
                context_ticket = context.get('ticket', '#0')
                event_ticket = event.get('ticket', '#0')
                ticket_str = context_ticket if context_ticket != '#0' else event_ticket
                ticket_num = int(ticket_str.replace('#', '')) if ticket_str.startswith('#') else 0
                
                promotions.append({
                    'ticket': ticket_num,
                    'agent': context.get('agent', 'unknown'),
                    'timestamp': event.get('timestamp'),
                    'previous_status': interpretation.get('previous_status', 'unknown'),
                    'promotion_reason': context.get('promotion_reason', 'unknown')
                })
    
    return sorted(promotions, key=lambda x: x['ticket'])


def detect_alliance_patterns(main_logs: List[Dict], time_window: int = 3) -> List[Dict]:
    """Detect coordinated voting patterns (alliances).
    
    Args:
        main_logs: List of main log events
        time_window: Maximum ticket distance for alliance detection
        
    Returns:
        List of alliance events with participants and timing
    """
    vote_events = []
    
    # Extract all vote events using documented pattern
    for event in main_logs:
        if (event.get('event_type') == 'SYMBOL_INTERPRETATION' and
            event.get('details', {}).get('symbol_type') == 'VOTE'):
            
            context = event.get('details', {}).get('context', {})
            interpretation = event.get('details', {}).get('interpretation', {})
            
            # Extract ticket number
            ticket_str = context.get('ticket', '#0')
            ticket_num = int(ticket_str.replace('#', '')) if ticket_str.startswith('#') else 0
            
            vote_events.append({
                'ticket': ticket_num,
                'voter': context.get('author', 'unknown'),
                'target': interpretation.get('target', 'unknown'),
                'timestamp': event.get('timestamp')
            })
    
    # Group by target within time windows
    alliances = []
    for target in set(vote['target'] for vote in vote_events):
        target_votes = [v for v in vote_events if v['target'] == target]
        target_votes.sort(key=lambda x: x['ticket'])
        
        # Detect clusters within time_window
        current_cluster = []
        for vote in target_votes:
            if not current_cluster or vote['ticket'] - current_cluster[-1]['ticket'] <= time_window:
                current_cluster.append(vote)
            else:
                if len(current_cluster) >= 2:  # Alliance = 2+ coordinated votes
                    alliances.append({
                        'target': target,
                        'participants': [v['voter'] for v in current_cluster],
                        'tickets': [v['ticket'] for v in current_cluster],
                        'duration': max(v['ticket'] for v in current_cluster) - min(v['ticket'] for v in current_cluster),
                        'start_ticket': min(v['ticket'] for v in current_cluster),
                        'end_ticket': max(v['ticket'] for v in current_cluster)
                    })
                current_cluster = [vote]
        
        # Handle final cluster
        if len(current_cluster) >= 2:
            alliances.append({
                'target': target,
                'participants': [v['voter'] for v in current_cluster],
                'tickets': [v['ticket'] for v in current_cluster],
                'duration': max(v['ticket'] for v in current_cluster) - min(v['ticket'] for v in current_cluster),
                'start_ticket': min(v['ticket'] for v in current_cluster),
                'end_ticket': max(v['ticket'] for v in current_cluster)
            })
    
    return alliances


def analyze_immune_stabilization(metrics_logs: List[Dict]) -> List[Dict]:
    """Analyze pressure detection → stabilization timing from metrics logs.
    
    Args:
        metrics_logs: List of metrics log events
        
    Returns:
        List of immune adjustment events with timing and agent details
    """
    immune_events = []
    
    for event in metrics_logs:
        if event.get('event_type') == 'IMMUNE_RESPONSE_ADJUSTMENT':
            details = event.get('details', {})
            agent_details = details.get('agent_details', {})
            
            # Extract frequency changes for all agents
            agent_adjustments = {}
            for agent, agent_info in agent_details.items():
                agent_adjustments[agent] = {
                    'frequency_before': agent_info.get('frequency_before', 0),
                    'frequency_after': agent_info.get('frequency_after', 0),
                    'adjustment_reason': agent_info.get('adjustment_reason', 'unknown'),
                    'safety_cap_applied': agent_info.get('safety_cap_applied', False)
                }
            
            immune_events.append({
                'ticket': details.get('ticket', 0),
                'pressure_level': details.get('pressure_level', 'UNKNOWN'),
                'recent_promotions': details.get('recent_promotions', 0),
                'monitoring_window': details.get('monitoring_window', 12),
                'agents_adjusted': details.get('agents_adjusted', []),
                'agent_adjustments': agent_adjustments,
                'reduction_multiplier': details.get('reduction_multiplier_used'),
                'boost_multiplier': details.get('boost_multiplier_used'),
                'memory_dampening': details.get('memory_dampening_applied', False),
                'timestamp': event.get('timestamp')
            })
    
    return sorted(immune_events, key=lambda x: x['ticket'])


def extract_vote_patterns(main_logs: List[Dict]) -> List[Dict]:
    """Extract vote processing events with weights and multipliers.
    
    Args:
        main_logs: List of main log events
        
    Returns:
        List of vote events with detailed weight information
    """
    votes = []
    
    for event in main_logs:
        if event.get('event_type') == 'VOTE_PROCESSING':
            details = event.get('details', {})
            
            votes.append({
                'voter': details.get('voter', 'unknown'),
                'target': details.get('target', 'unknown'),
                'vote_value': details.get('value', 1),
                'actual_value': details.get('actual_value', 1),
                'is_self_vote': details.get('is_self_vote', False),
                'is_binder_vote': details.get('is_binder_vote', False),
                'binder_multiplier': details.get('binder_multiplier'),
                'self_vote_weight': details.get('self_vote_weight'),
                'ticket': details.get('ticket', '#0'),
                'message_id': details.get('message_id', 0),
                'timestamp': event.get('timestamp')
            })
    
    return votes


def extract_gate_decisions(main_logs: List[Dict]) -> List[Dict]:
    """Extract gate decision events for security analysis.
    
    Args:
        main_logs: List of main log events
        
    Returns:
        List of gate decision events
    """
    gates = []
    
    for event in main_logs:
        if event.get('event_type') == 'GATE_DECISION':
            details = event.get('details', {})
            
            gates.append({
                'gate_type': details.get('gate', 'unknown'),
                'result': details.get('result', 'unknown'),
                'reason': details.get('reason', ''),
                'content_hash': details.get('content_hash', ''),
                'detection_method': details.get('detection_method', ''),
                'max_score': details.get('max_score'),
                'timestamp': event.get('timestamp')
            })
    
    return gates


def validate_log_parsing(batch_dir: Path) -> Dict[str, int]:
    """Validate parsing against known batch structure.
    
    Args:
        batch_dir: Path to batch directory
        
    Returns:
        Dictionary with parsing statistics
    """
    main_logs, debug_logs = load_experiment_logs(batch_dir)
    
    # Count different event types
    vote_events = [e for e in main_logs if e.get('event_type') == 'SYMBOL_INTERPRETATION' 
                   and e.get('details', {}).get('symbol_type') == 'VOTE']
    
    binder_promotions = [e for e in main_logs if e.get('event_type') == 'SYMBOL_INTERPRETATION'
                        and e.get('details', {}).get('symbol_type') == 'STATUS_CHANGE'
                        and e.get('details', {}).get('interpretation', {}).get('new_status') == 'BINDER']
    
    immune_adjustments = [e for e in debug_logs if e.get('event_type') == 'IMMUNE_RESPONSE_ADJUSTMENT']
    
    gate_decisions = [e for e in main_logs if e.get('event_type') == 'GATE_DECISION']
    
    return {
        'main_events': len(main_logs),
        'debug_events': len(debug_logs),
        'vote_events': len(vote_events),
        'binder_promotions': len(binder_promotions),
        'immune_adjustments': len(immune_adjustments),
        'gate_decisions': len(gate_decisions)
    }


if __name__ == "__main__":
    # Example usage and validation
    batch_dir = Path("exp_output/published_data/batch_11_adaptive_immune")
    if batch_dir.exists():
        print("Testing data parsing functions...")
        
        # Load logs
        main_logs, debug_logs = load_experiment_logs(batch_dir)
        print(f"Loaded {len(main_logs)} main events, {len(debug_logs)} debug events")
        
        # Extract data
        promotions = extract_binder_promotions(main_logs)
        alliances = detect_alliance_patterns(main_logs)
        immune_events = analyze_immune_stabilization(debug_logs)
        
        print(f"Found {len(promotions)} BINDER promotions")
        print(f"Found {len(alliances)} alliance formations")
        print(f"Found {len(immune_events)} immune adjustments")
        
        # Validate structure
        validation = validate_log_parsing(batch_dir)
        print(f"Validation results: {validation}")
    else:
        print(f"Batch directory {batch_dir} not found")