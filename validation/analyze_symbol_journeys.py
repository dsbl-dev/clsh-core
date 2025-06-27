#!/usr/bin/env python3
"""
Symbol Journey Timeline Analyzer for experiments.
Analyzes how symbol meanings evolve over time based on context changes.
"""

import json
import sys
import pathlib
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse

def load_experiment_log(log_file: pathlib.Path) -> List[Dict]:
    """Load JSONL experiment log file."""
    events = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                event = json.loads(line.strip())
                events.append(event)
            except json.JSONDecodeError as e:
                print(f"Warning: JSON error on line {line_num}: {e}")
                continue
    return events

def extract_symbol_interpretations(events: List[Dict]) -> List[Dict]:
    """Extract all SYMBOL_INTERPRETATION events from log."""
    return [e for e in events if e.get('event_type') == 'SYMBOL_INTERPRETATION']

def extract_status_changes(events: List[Dict]) -> List[Dict]:
    """Extract all STATUS_CHANGE events from log."""
    status_changes = []
    for event in events:
        # Check for direct STATUS_CHANGE events
        if event.get('event_type') == 'STATUS_CHANGE':
            status_changes.append(event)
        # Check for SYMBOL_INTERPRETATION with STATUS_CHANGE symbol_type
        elif (event.get('event_type') == 'SYMBOL_INTERPRETATION' and 
              event.get('details', {}).get('symbol_type') == 'STATUS_CHANGE'):
            status_changes.append(event)
        # Check for MAA Immune System events (v2.11)
        elif event.get('event_type') in ['IMMUNE_FREQUENCY_ADJUSTMENT', 'ADAPTIVE_ADJUSTMENT', 'IMMUNE_RESPONSE_ADJUSTMENT']:
            status_changes.append(event)
    return status_changes

def analyze_symbol_journey(symbol_interpretations: List[Dict], status_changes: List[Dict] = None) -> Dict:
    """Analyze how symbols evolve their meanings over time."""
    
    # Group interpretations by symbol content
    symbol_journeys = defaultdict(list)
    
    for interpretation in symbol_interpretations:
        details = interpretation.get('details', {})
        symbol_content = details.get('symbol_content')
        ticket = details.get('context', {}).get('ticket', '#0')
        
        if symbol_content:
            symbol_journeys[symbol_content].append({
                'ticket': ticket,
                'timestamp': interpretation.get('timestamp'),
                'interpreter': details.get('interpreter'),
                'context': details.get('context', {}),
                'interpretation': details.get('interpretation', {}),
                'symbol_type': details.get('symbol_type')
            })
    
    # Sort each journey by ticket number
    for symbol, journey in symbol_journeys.items():
        journey.sort(key=lambda x: int(x['ticket'].replace('#', '')) if x['ticket'].startswith('#') else 0)
    
    analysis = {
        'total_symbols_tracked': len(symbol_journeys),
        'total_interpretations': len(symbol_interpretations),
        'symbol_journeys': dict(symbol_journeys),
        'symbol_evolution_patterns': analyze_evolution_patterns(symbol_journeys),
        'interpretation_frequency': analyze_interpretation_frequency(symbol_interpretations),
        'context_impact': analyze_context_impact(symbol_interpretations),
        'binder_analysis': analyze_binder_patterns(symbol_interpretations, status_changes) if status_changes else None,
        'production_insights': analyze_production_patterns(symbol_journeys),
        'adaptive_immune_analysis': analyze_adaptive_immune_patterns(symbol_interpretations, status_changes) if status_changes else None,
        'visualization_timeline_data': export_timeline_visualization_data(symbol_journeys, status_changes)
    }
    
    return analysis

def analyze_evolution_patterns(symbol_journeys: Dict[str, List[Dict]]) -> Dict:
    """Analyze patterns in how symbols evolve."""
    patterns = {
        'semantic_shifts': [],  # When interpretation changes for same symbol
        'context_dependencies': [],  # When context affects interpretation
        'vote_weight_evolution': [],  # Track vote weight changes over time
        'status_dependent_changes': []  # Changes based on user status
    }
    
    for symbol, journey in symbol_journeys.items():
        if len(journey) < 2:
            continue
            
        # Analyze semantic shifts
        prev_interpretation = None
        for event in journey:
            current_interpretation = event['interpretation']
            
            if prev_interpretation and current_interpretation != prev_interpretation:
                patterns['semantic_shifts'].append({
                    'symbol': symbol,
                    'from_interpretation': prev_interpretation,
                    'to_interpretation': current_interpretation,
                    'ticket': event['ticket'],
                    'context_change': compare_contexts(journey[journey.index(event)-1]['context'], event['context'])
                })
            
            prev_interpretation = current_interpretation
        
        # Analyze vote weight evolution for VOTE symbols
        if symbol.startswith(('promote_', 'demote_')):
            weight_changes = []
            for event in journey:
                interpretation = event['interpretation']
                if 'reputation_multiplier' in interpretation:
                    weight_changes.append({
                        'ticket': event['ticket'],
                        'multiplier': interpretation['reputation_multiplier'],
                        'target_reputation': event['context'].get('target_reputation')
                    })
            
            if weight_changes:
                patterns['vote_weight_evolution'].append({
                    'symbol': symbol,
                    'weight_timeline': weight_changes
                })
    
    return patterns

def compare_contexts(context1: Dict, context2: Dict) -> Dict:
    """Compare two contexts to identify what changed."""
    changes = {}
    
    # Check for status changes
    if context1.get('old_status') != context2.get('old_status'):
        changes['status_change'] = {
            'from': context1.get('old_status'),
            'to': context2.get('old_status')
        }
    
    # Check for reputation changes
    rep1 = context1.get('target_reputation') or context1.get('user_reputation', 0)
    rep2 = context2.get('target_reputation') or context2.get('user_reputation', 0)
    if abs(rep1 - rep2) > 0.1:
        changes['reputation_change'] = {
            'from': rep1,
            'to': rep2,
            'delta': rep2 - rep1
        }
    
    # Check for ticket progression
    ticket1 = context1.get('ticket', '#0')
    ticket2 = context2.get('ticket', '#0')
    if ticket1 != ticket2:
        changes['ticket_progression'] = {
            'from': ticket1,
            'to': ticket2
        }
    
    return changes

def analyze_interpretation_frequency(symbol_interpretations: List[Dict]) -> Dict:
    """Analyze frequency of different interpretation types."""
    
    frequency_analysis = {
        'by_symbol_type': Counter(),
        'by_interpreter': Counter(),
        'by_action': Counter(),
        'timeline_distribution': defaultdict(int)
    }
    
    for interpretation in symbol_interpretations:
        details = interpretation.get('details', {})
        
        # Count by symbol type
        symbol_type = details.get('symbol_type', 'unknown')
        frequency_analysis['by_symbol_type'][symbol_type] += 1
        
        # Count by interpreter
        interpreter = details.get('interpreter', 'unknown')
        frequency_analysis['by_interpreter'][interpreter] += 1
        
        # Count by action
        action = details.get('interpretation', {}).get('action', 'unknown')
        frequency_analysis['by_action'][action] += 1
        
        # Timeline distribution (by ticket ranges)
        ticket = details.get('context', {}).get('ticket', '#0')
        try:
            ticket_num = int(ticket.replace('#', ''))
            ticket_range = f"{(ticket_num // 10) * 10}-{(ticket_num // 10) * 10 + 9}"
            frequency_analysis['timeline_distribution'][ticket_range] += 1
        except (ValueError, AttributeError):
            frequency_analysis['timeline_distribution']['unknown'] += 1
    
    # Convert to regular dicts for JSON serialization
    frequency_analysis['by_symbol_type'] = dict(frequency_analysis['by_symbol_type'])
    frequency_analysis['by_interpreter'] = dict(frequency_analysis['by_interpreter'])
    frequency_analysis['by_action'] = dict(frequency_analysis['by_action'])
    frequency_analysis['timeline_distribution'] = dict(frequency_analysis['timeline_distribution'])
    
    return frequency_analysis

def analyze_context_impact(symbol_interpretations: List[Dict]) -> Dict:
    """Analyze how context affects symbol interpretation."""
    
    context_impact = {
        'reputation_effects': [],
        'status_effects': [],
        'temporal_effects': []
    }
    
    # Group interpretations by symbol content
    symbol_groups = defaultdict(list)
    for interpretation in symbol_interpretations:
        details = interpretation.get('details', {})
        symbol_content = details.get('symbol_content')
        if symbol_content:
            symbol_groups[symbol_content].append(interpretation)
    
    # Analyze context effects for each symbol
    for symbol, interpretations in symbol_groups.items():
        if len(interpretations) < 2:
            continue
            
        # Analyze reputation effects
        reputation_interpretations = []
        for interp in interpretations:
            details = interp.get('details', {})
            context = details.get('context', {})
            interpretation = details.get('interpretation', {})
            
            target_rep = context.get('target_reputation')
            reputation_mult = interpretation.get('reputation_multiplier')
            
            if target_rep is not None and reputation_mult is not None:
                reputation_interpretations.append({
                    'reputation': target_rep,
                    'multiplier': reputation_mult,
                    'ticket': context.get('ticket')
                })
        
        if reputation_interpretations:
            context_impact['reputation_effects'].append({
                'symbol': symbol,
                'reputation_curve': reputation_interpretations
            })
    
    return context_impact

def analyze_binder_patterns(symbol_interpretations: List[Dict], status_changes: List[Dict]) -> Dict:
    """Analyze patterns specific to BINDER emergence and post-promotion behavior."""
    
    binder_analysis = {
        'promotions_detected': [],
        'pre_promotion_patterns': {},
        'post_promotion_changes': {},
        'vote_weight_evolution': [],
        'territorial_patterns': {}
    }
    
    # Extract BINDER promotions from status changes
    binder_promotions = []
    for change in status_changes:
        details = change.get('details', {})
        interpretation = details.get('interpretation', {})
        context = details.get('context', {})
        
        # Check both direct new_status and interpretation new_status
        new_status = details.get('new_status') or interpretation.get('new_status')
        if new_status == 'BINDER':
            user = details.get('user') or context.get('username')
            ticket = details.get('ticket') or context.get('ticket')
            vote_count = details.get('final_vote_count') or context.get('vote_count', 0)
            
            binder_promotions.append({
                'user': user,
                'ticket': ticket,
                'timestamp': change.get('timestamp'),
                'promotion_vote_count': vote_count
            })
    
    binder_analysis['promotions_detected'] = binder_promotions
    
    # Analyze pre vs post promotion symbol usage for each BINDER
    for promotion in binder_promotions:
        user = promotion['user']
        promotion_ticket = int(promotion['ticket'].replace('#', ''))
        
        # Split interpretations into pre/post promotion
        pre_promotion = []
        post_promotion = []
        
        for interp in symbol_interpretations:
            details = interp.get('details', {})
            context = details.get('context', {})
            ticket_str = context.get('ticket', '#0')
            
            try:
                ticket_num = int(ticket_str.replace('#', ''))
                author = context.get('author')
                
                if author == user:
                    if ticket_num < promotion_ticket:
                        pre_promotion.append(interp)
                    elif ticket_num >= promotion_ticket:
                        post_promotion.append(interp)
            except (ValueError, AttributeError):
                continue
        
        binder_analysis['pre_promotion_patterns'][user] = {
            'symbol_count': len(pre_promotion),
            'symbols_used': list(set(d.get('details', {}).get('symbol_content') for d in pre_promotion))
        }
        
        binder_analysis['post_promotion_changes'][user] = {
            'symbol_count': len(post_promotion),
            'symbols_used': list(set(d.get('details', {}).get('symbol_content') for d in post_promotion)),
            'vote_multiplier_usage': len([d for d in post_promotion 
                                        if d.get('details', {}).get('interpretation', {}).get('semantic_change')])
        }
    
    return binder_analysis

def analyze_production_patterns(symbol_journeys: Dict[str, List[Dict]]) -> Dict:
    """Analyze patterns specific to production dataset (40 runs × 60 tickets)."""
    
    production_patterns = {
        'vote_effectiveness_patterns': [],
        'reputation_weighting_effects': [],
        'civil_gate_correlations': [],
        'extended_timeline_effects': {},
        'duplicate_vote_patterns': []
    }
    
    # Analyze reputation weighting effects (key production insight)
    for symbol, journey in symbol_journeys.items():
        if symbol.startswith(('promote_', 'demote_')):
            weighted_votes = []
            for event in journey:
                interpretation = event['interpretation']
                context = event['context']
                
                # Check for reputation weighting
                if 'reputation_multiplier' in interpretation:
                    weighted_votes.append({
                        'symbol': symbol,
                        'ticket': event['ticket'],
                        'original_vote': context.get('vote_value', 1),
                        'multiplier': interpretation['reputation_multiplier'],
                        'weighted_value': interpretation.get('weighted_vote_value'),
                        'target_reputation': context.get('target_reputation'),
                        'semantic_change': interpretation.get('semantic_change', '')
                    })
            
            if weighted_votes:
                production_patterns['reputation_weighting_effects'].append({
                    'symbol': symbol,
                    'weighted_instances': weighted_votes,
                    'effectiveness_reduction': sum(1 - w['multiplier'] for w in weighted_votes if w['multiplier'] < 1.0)
                })
    
    # Analyze extended timeline effects (60 tickets vs previous 40)
    ticket_ranges = {
        'early': (1, 20),
        'middle': (21, 40), 
        'extended': (41, 60)
    }
    
    for range_name, (start, end) in ticket_ranges.items():
        range_symbols = []
        for symbol, journey in symbol_journeys.items():
            for event in journey:
                try:
                    ticket_num = int(event['ticket'].replace('#', ''))
                    if start <= ticket_num <= end:
                        range_symbols.append(symbol)
                        break
                except (ValueError, AttributeError):
                    continue
        
        production_patterns['extended_timeline_effects'][range_name] = {
            'unique_symbols': len(set(range_symbols)),
            'total_activity': len(range_symbols),
            'symbol_diversity': len(set(range_symbols)) / max(len(range_symbols), 1)
        }
    
    return production_patterns

def analyze_adaptive_immune_patterns(symbol_interpretations: List[Dict], status_changes: List[Dict]) -> Dict:
    """Analyze patterns specific to MAA Immune System v2.11."""
    
    adaptive_patterns = {
        'frequency_adjustments': [],
        'multi_agent_coordination': {},
        'dave_zara_emergence': {},
        'adaptive_response_timeline': [],
        'coordinated_pressure_events': []
    }
    
    # Extract frequency adjustment events
    frequency_adjustments = [e for e in status_changes 
                           if e.get('event_type') in ['IMMUNE_FREQUENCY_ADJUSTMENT', 'ADAPTIVE_ADJUSTMENT', 'IMMUNE_RESPONSE_ADJUSTMENT']]
    
    for adjustment in frequency_adjustments:
        details = adjustment.get('details', {})
        
        # Handle IMMUNE_RESPONSE_ADJUSTMENT format
        if adjustment.get('event_type') == 'IMMUNE_RESPONSE_ADJUSTMENT':
            agent_details = details.get('agent_details', {})
            for agent_name, agent_data in agent_details.items():
                adaptive_patterns['frequency_adjustments'].append({
                    'timestamp': adjustment.get('timestamp'),
                    'agent': agent_name,
                    'old_frequency': agent_data.get('frequency_before'),
                    'new_frequency': agent_data.get('frequency_after'),
                    'pressure_level': details.get('pressure_level'),
                    'coordination_type': 'multi_agent',
                    'adjustment_reason': agent_data.get('adjustment_reason'),
                    'ticket': details.get('ticket')
                })
        else:
            # Handle legacy format
            adaptive_patterns['frequency_adjustments'].append({
                'timestamp': adjustment.get('timestamp'),
                'agent': details.get('agent'),
                'old_frequency': details.get('old_frequency'),
                'new_frequency': details.get('new_frequency'),
                'pressure_level': details.get('pressure_level'),
                'coordination_type': details.get('coordination_type', 'individual')
            })
    
    # Analyze multi-agent coordination
    agent_adjustments = {}
    for adj in adaptive_patterns['frequency_adjustments']:
        agent = adj['agent']
        if agent not in agent_adjustments:
            agent_adjustments[agent] = []
        agent_adjustments[agent].append(adj)
    
    adaptive_patterns['multi_agent_coordination'] = {
        'eve_adjustments': len(agent_adjustments.get('eve', [])),
        'dave_adjustments': len(agent_adjustments.get('dave', [])),
        'zara_adjustments': len(agent_adjustments.get('zara', [])),
        'synchronized_events': count_synchronized_adjustments(agent_adjustments)
    }
    
    return adaptive_patterns

def count_synchronized_adjustments(agent_adjustments: Dict) -> int:
    """Count coordinated frequency adjustments between agents."""
    synchronized_count = 0
    
    eve_times = [adj['timestamp'] for adj in agent_adjustments.get('eve', [])]
    dave_times = [adj['timestamp'] for adj in agent_adjustments.get('dave', [])]
    zara_times = [adj['timestamp'] for adj in agent_adjustments.get('zara', [])]
    
    # For same-ticket adjustments (multi-agent coordination), count as synchronized
    eve_tickets = [adj.get('ticket') for adj in agent_adjustments.get('eve', [])]
    dave_tickets = [adj.get('ticket') for adj in agent_adjustments.get('dave', [])]
    zara_tickets = [adj.get('ticket') for adj in agent_adjustments.get('zara', [])]
    
    # Count tickets that appear in multiple agent adjustments (coordinated responses)
    all_tickets = eve_tickets + dave_tickets + zara_tickets
    ticket_counts = {}
    for ticket in all_tickets:
        if ticket:
            ticket_counts[ticket] = ticket_counts.get(ticket, 0) + 1
    
    synchronized_count = sum(1 for count in ticket_counts.values() if count > 1)
    
    return synchronized_count

def generate_symbol_timeline_report(analysis: Dict, output_file: Optional[pathlib.Path] = None) -> str:
    """Generate human-readable report of symbol journey analysis."""
    
    report = []
    report.append("# Symbol Journey Timeline Analysis Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Summary statistics
    report.append("## Summary Statistics")
    report.append(f"- Total symbols tracked: {analysis['total_symbols_tracked']}")
    report.append(f"- Total interpretations logged: {analysis['total_interpretations']}")
    report.append("")
    
    # Interpretation frequency
    freq = analysis['interpretation_frequency']
    report.append("## Interpretation Frequency")
    report.append("### By Symbol Type:")
    for symbol_type, count in freq['by_symbol_type'].items():
        report.append(f"- {symbol_type}: {count}")
    
    report.append("\n### By Interpreter:")
    for interpreter, count in freq['by_interpreter'].items():
        report.append(f"- {interpreter}: {count}")
    
    report.append("\n### By Action:")
    for action, count in freq['by_action'].items():
        report.append(f"- {action}: {count}")
    
    # Timeline distribution
    report.append("\n### Timeline Distribution (by ticket ranges):")
    timeline_items = sorted(freq['timeline_distribution'].items(), 
                           key=lambda x: int(x[0].split('-')[0]) if '-' in x[0] and x[0].split('-')[0].isdigit() else 999)
    for ticket_range, count in timeline_items:
        report.append(f"- Tickets {ticket_range}: {count} interpretations")
    
    # Evolution patterns
    patterns = analysis['symbol_evolution_patterns']
    report.append("\n## Symbol Evolution Patterns")
    
    if patterns['semantic_shifts']:
        report.append("### Semantic Shifts Detected:")
        for shift in patterns['semantic_shifts'][:10]:  # Show top 10
            report.append(f"- **{shift['symbol']}** at {shift['ticket']}: {shift['from_interpretation'].get('action', 'unknown')} → {shift['to_interpretation'].get('action', 'unknown')}")
    
    if patterns['vote_weight_evolution']:
        report.append("\n### Vote Weight Evolution:")
        for evolution in patterns['vote_weight_evolution'][:5]:  # Show top 5
            symbol = evolution['symbol']
            timeline = evolution['weight_timeline']
            report.append(f"- **{symbol}**: {len(timeline)} weight changes")
            for change in timeline[:3]:  # Show first 3 changes
                rep = change.get('target_reputation', 'unknown')
                mult = change.get('multiplier', 1.0)
                report.append(f"  - {change['ticket']}: {mult:.2f}x multiplier (target rep: {rep})")
    
    # BINDER analysis (production insight)
    binder_analysis = analysis.get('binder_analysis')
    if binder_analysis:
        report.append("\n## BINDER Emergence Analysis")
        promotions = binder_analysis['promotions_detected']
        report.append(f"### BINDER Promotions Detected: {len(promotions)}")
        
        for promotion in promotions:
            user = promotion['user']
            ticket = promotion['ticket']
            vote_count = promotion['promotion_vote_count']
            report.append(f"- **{user}** promoted at {ticket} (final vote count: {vote_count})")
            
            # Pre vs post patterns
            pre = binder_analysis['pre_promotion_patterns'].get(user, {})
            post = binder_analysis['post_promotion_changes'].get(user, {})
            
            report.append(f"  - Pre-promotion: {pre.get('symbol_count', 0)} symbols")
            report.append(f"  - Post-promotion: {post.get('symbol_count', 0)} symbols, {post.get('vote_multiplier_usage', 0)} weighted votes")
    
    # Production patterns
    production = analysis.get('production_insights')
    if production:
        report.append("\n## Production Dataset Insights (40 runs × 60 tickets)")
        
        # Reputation weighting effects
        weighting_effects = production['reputation_weighting_effects']
        if weighting_effects:
            report.append(f"### Reputation Weighting Effects: {len(weighting_effects)} symbols affected")
            for effect in weighting_effects[:3]:
                symbol = effect['symbol']
                instances = len(effect['weighted_instances'])
                reduction = effect['effectiveness_reduction']
                report.append(f"- **{symbol}**: {instances} weighted votes, {reduction:.2f} total effectiveness reduction")
        
        # Extended timeline effects
        timeline_effects = production['extended_timeline_effects']
        report.append("\n### Extended Timeline Analysis (60-ticket experiments):")
        for period, stats in timeline_effects.items():
            diversity = stats['symbol_diversity']
            unique = stats['unique_symbols']
            total = stats['total_activity']
            report.append(f"- **{period.title()} period**: {unique} unique symbols, {total} total activity, {diversity:.2f} diversity")
    
    # MAA Immune System analysis (v2.11)
    adaptive_analysis = analysis.get('adaptive_immune_analysis')
    if adaptive_analysis:
        report.append("\n## MAA Immune System Analysis (v2.11)")
        
        coordination = adaptive_analysis['multi_agent_coordination']
        report.append(f"### Agent Frequency Adjustments:")
        report.append(f"- **Eve**: {coordination['eve_adjustments']} adjustments")
        report.append(f"- **Dave**: {coordination['dave_adjustments']} adjustments") 
        report.append(f"- **Zara**: {coordination['zara_adjustments']} adjustments")
        report.append(f"- **Synchronized Events**: {coordination['synchronized_events']} coordinated responses")
        
        if adaptive_analysis['frequency_adjustments']:
            report.append("\n### Recent Frequency Adjustments:")
            for adj in adaptive_analysis['frequency_adjustments'][:5]:  # Show last 5
                agent = adj['agent']
                old_freq = adj['old_frequency']
                new_freq = adj['new_frequency'] 
                pressure = adj['pressure_level']
                report.append(f"- **{agent}**: {old_freq} → {new_freq} (pressure: {pressure})")
    
    # Context impact summary
    context = analysis['context_impact']
    report.append("\n## Context Impact Summary")
    
    reputation_effects = len(context['reputation_effects'])
    if reputation_effects > 0:
        report.append(f"- {reputation_effects} symbols showed reputation-dependent interpretation changes")
    
    # Example journeys
    report.append("\n## Example Symbol Journeys")
    journeys = analysis['symbol_journeys']
    example_count = 0
    
    for symbol, journey in journeys.items():
        if example_count >= 3:  # Show 3 examples
            break
        if len(journey) >= 3:  # Only show interesting journeys
            report.append(f"\n### {symbol}")
            for i, event in enumerate(journey[:5]):  # Show first 5 events
                action = event['interpretation'].get('action', 'unknown')
                interpreter = event.get('interpreter', 'unknown')
                ticket = event.get('ticket', '#0')
                report.append(f"{i+1}. {ticket} ({interpreter}): {action}")
            example_count += 1
    
    report_text = "\n".join(report)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"Symbol timeline report saved to: {output_file}")
    
    return report_text

def main():
    parser = argparse.ArgumentParser(description="Analyze Symbol Journey Timeline from experiment logs")
    parser.add_argument("log_files", nargs="+", help="JSONL log files to analyze")
    parser.add_argument("--output", "-o", help="Output file for report")
    parser.add_argument("--json", action="store_true", help="Output detailed JSON analysis")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("Symbol Journey Timeline Analyzer")
    print("=" * 60)
    
    all_symbol_interpretations = []
    all_status_changes = []
    
    # Load all log files (including metrics logs)
    for log_file_str in args.log_files:
        log_file = pathlib.Path(log_file_str)
        if not log_file.exists():
            print(f"File not found: {log_file}")
            continue
            
        print(f"Loading: {log_file.name}")
        events = load_experiment_log(log_file)
        
        symbol_interpretations = extract_symbol_interpretations(events)
        status_changes = extract_status_changes(events)
        
        all_symbol_interpretations.extend(symbol_interpretations)
        all_status_changes.extend(status_changes)
        
        # Also load corresponding metrics log for immune system data
        metrics_name = log_file.stem.replace("_d", "_metrics_d") + ".jsonl"
        metrics_path = log_file.parent / "metrics" / metrics_name
        
        if metrics_path.exists():
            print(f"Loading metrics: {metrics_path.name}")
            metrics_events = load_experiment_log(metrics_path)
            metrics_status_changes = extract_status_changes(metrics_events)
            all_status_changes.extend(metrics_status_changes)
        
        if args.verbose:
            print(f"   - {len(events)} total events")
            print(f"   - {len(symbol_interpretations)} symbol interpretations")
            print(f"   - {len(status_changes)} status changes")
    
    if not all_symbol_interpretations:
        print("No symbol interpretations found in log files")
        return 1
    
    print(f"\nAnalyzing {len(all_symbol_interpretations)} symbol interpretations...")
    
    # Perform analysis
    analysis = analyze_symbol_journey(all_symbol_interpretations, all_status_changes)
    
    # Generate report
    output_file = pathlib.Path(args.output) if args.output else None
    report = generate_symbol_timeline_report(analysis, output_file)
    
    if not args.output:
        print("\n" + report)
    
    # Output JSON if requested
    if args.json:
        json_file = pathlib.Path(args.output).with_suffix('.json') if args.output else pathlib.Path("symbol_journey_analysis.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        print(f"Detailed JSON analysis saved to: {json_file}")
    
    print(f"\nAnalysis complete! Tracked {analysis['total_symbols_tracked']} unique symbols")
    
    # Export visualization timeline data for Phase 1 implementation
    if args.json and 'visualization_timeline_data' in analysis:
        if args.output:
            base_name = pathlib.Path(args.output).stem
            viz_file = pathlib.Path(args.output).parent / f"{base_name}_timeline.json"
        else:
            viz_file = pathlib.Path("symbol_timeline_visualization.json")
        with open(viz_file, 'w', encoding='utf-8') as f:
            json.dump(analysis['visualization_timeline_data'], f, indent=2, ensure_ascii=False, default=str)
        print(f"Timeline visualization data exported to: {viz_file}")
    
    return 0

def load_and_process_logs(log_files):
    """
    Load and process multiple log files for batch analysis.
    Includes both main logs and metrics logs for complete immune system data.
    Returns: (symbol_data, promotions, summary)
    """
    all_symbol_interpretations = []
    all_status_changes = []
    
    for log_file in log_files:
        log_path = pathlib.Path(log_file)
        if log_path.exists():
            try:
                events = load_experiment_log(log_path)
                symbol_interpretations = extract_symbol_interpretations(events)
                status_changes = extract_status_changes(events)
                
                all_symbol_interpretations.extend(symbol_interpretations)
                all_status_changes.extend(status_changes)
                
                # Also check for corresponding metrics
                metrics_name = log_path.stem.replace("_d", "_metrics_d") + ".jsonl"
                metrics_path = log_path.parent / "metrics" / metrics_name
                
                if metrics_path.exists():
                    metrics_events = load_experiment_log(metrics_path)
                    metrics_status_changes = extract_status_changes(metrics_events)
                    all_status_changes.extend(metrics_status_changes)
                    
            except Exception as e:
                print(f"Error processing {log_file}: {e}")
                continue
    
    # Analyze data
    analysis = analyze_symbol_journey(all_symbol_interpretations, all_status_changes)
    
    # Extract BINDER promotions for compatibility with analyze_validation_batch.py
    promotions = []
    for change in all_status_changes:
        if change.get('details', {}).get('type') == 'PROMOTION':
            promotions.append({
                'user': change['details']['username'],
                'ticket': change['details'].get('ticket', 'unknown'),
                'timestamp': change.get('timestamp', 'unknown')
            })
    
    # Format symbol data for compatibility
    symbol_data = {}
    for interp in all_symbol_interpretations:
        symbol = interp.get('details', {}).get('symbol_content', 'unknown')
        if symbol not in symbol_data:
            symbol_data[symbol] = []
        symbol_data[symbol].append(interp)
    
    # Summary data
    summary = {
        'total_interpretations': len(all_symbol_interpretations),
        'total_promotions': len(promotions),
        'unique_symbols': len(symbol_data),
        'files_processed': len([f for f in log_files if pathlib.Path(f).exists()])
    }
    
    return symbol_data, promotions, summary


def export_timeline_visualization_data(symbol_journeys: Dict[str, List[Dict]], status_changes: List[Dict] = None) -> Dict:
    """Export timeline coordinate data ready for visualization."""
    
    timeline_data = {
        'timeline_coordinates': [],
        'context_events': [],
        'vote_weight_evolution': [],
        'immune_frequency_timeline': [],
        'symbol_impact_matrix': {}
    }
    
    # Process each symbol journey into timeline coordinates
    for symbol, journey in symbol_journeys.items():
        if not journey:
            continue
            
        # Create timeline coordinates for this symbol
        symbol_timeline = {
            'symbol': symbol,
            'coordinates': [],
            'interpretation_changes': [],
            'vote_weight_changes': []
        }
        
        for event in journey:
            try:
                ticket_num = int(event['ticket'].replace('#', '')) if event['ticket'].startswith('#') else 0
                timestamp = event.get('timestamp', '')
                interpretation = event.get('interpretation', {})
                context = event.get('context', {})
                
                # Basic coordinate point
                coord_point = {
                    'x': ticket_num,  # X-axis: ticket progression
                    'y': calculate_symbol_impact_intensity(interpretation),  # Y-axis: meaning/impact intensity
                    'timestamp': timestamp,
                    'interpreter': event.get('interpreter'),
                    'context': context,
                    'interpretation': interpretation
                }
                
                symbol_timeline['coordinates'].append(coord_point)
                
                # Track interpretation changes
                if len(symbol_timeline['coordinates']) > 1:
                    prev_interp = symbol_timeline['coordinates'][-2]['interpretation']
                    if interpretation != prev_interp:
                        symbol_timeline['interpretation_changes'].append({
                            'ticket': ticket_num,
                            'from_interpretation': prev_interp.get('action', 'unknown'),
                            'to_interpretation': interpretation.get('action', 'unknown'),
                            'semantic_change': interpretation.get('semantic_change', '')
                        })
                
                # Track vote weight evolution for VOTE symbols
                if symbol.startswith(('promote_', 'demote_')) and 'reputation_multiplier' in interpretation:
                    symbol_timeline['vote_weight_changes'].append({
                        'ticket': ticket_num,
                        'multiplier': interpretation['reputation_multiplier'],
                        'weighted_value': interpretation.get('weighted_vote_value'),
                        'target_reputation': context.get('target_reputation')
                    })
                    
            except (ValueError, TypeError) as e:
                continue
        
        if symbol_timeline['coordinates']:
            timeline_data['timeline_coordinates'].append(symbol_timeline)
    
    # Process context events (status changes, immune responses)
    if status_changes:
        for change in status_changes:
            event_type = change.get('event_type')
            details = change.get('details', {})
            
            # BINDER promotions/demotions
            if event_type == 'STATUS_CHANGE' or (event_type == 'SYMBOL_INTERPRETATION' and details.get('symbol_type') == 'STATUS_CHANGE'):
                new_status = details.get('new_status') or details.get('interpretation', {}).get('new_status')
                if new_status:
                    timeline_data['context_events'].append({
                        'type': 'status_change',
                        'ticket': extract_ticket_number(details),
                        'timestamp': change.get('timestamp'),
                        'user': details.get('user') or details.get('context', {}).get('username'),
                        'new_status': new_status,
                        'event_details': details
                    })
            
            # Immune frequency adjustments (v2.11)
            elif event_type == 'IMMUNE_RESPONSE_ADJUSTMENT':
                agent_details = details.get('agent_details', {})
                for agent_name, agent_data in agent_details.items():
                    timeline_data['immune_frequency_timeline'].append({
                        'ticket': details.get('ticket', 0),
                        'timestamp': change.get('timestamp'),
                        'agent': agent_name,
                        'frequency_before': agent_data.get('frequency_before'),
                        'frequency_after': agent_data.get('frequency_after'),
                        'pressure_level': details.get('pressure_level'),
                        'adjustment_reason': agent_data.get('adjustment_reason')
                    })
                    
                timeline_data['context_events'].append({
                    'type': 'immune_adjustment',
                    'ticket': details.get('ticket', 0),
                    'timestamp': change.get('timestamp'),
                    'pressure_level': details.get('pressure_level'),
                    'agents_adjusted': list(agent_details.keys()),
                    'coordination_type': 'multi_agent'
                })
    
    # Generate symbol impact matrix (current state)
    timeline_data['symbol_impact_matrix'] = generate_symbol_impact_matrix(symbol_journeys)
    
    return timeline_data

def calculate_symbol_impact_intensity(interpretation: Dict) -> float:
    """Calculate Y-axis intensity value for symbol interpretation."""
    base_intensity = 1.0
    
    # Factor in vote weight multipliers
    if 'reputation_multiplier' in interpretation:
        multiplier = interpretation['reputation_multiplier']
        base_intensity *= multiplier
    
    # Factor in semantic changes
    if interpretation.get('semantic_change'):
        base_intensity += 0.5  # Boost for semantic evolution
    
    # Factor in action importance
    action = interpretation.get('action', '')
    if action in ['PROMOTE', 'DEMOTE']:
        base_intensity += 0.3
    elif action in ['BIND', 'CIVIL']:
        base_intensity += 0.2
    
    return base_intensity

def extract_ticket_number(details: Dict) -> int:
    """Extract ticket number from various detail formats."""
    # Try different possible locations for ticket info
    ticket_sources = [
        details.get('ticket'),
        details.get('context', {}).get('ticket'),
        details.get('ticket_id')
    ]
    
    for ticket in ticket_sources:
        if ticket:
            try:
                if isinstance(ticket, str) and ticket.startswith('#'):
                    return int(ticket.replace('#', ''))
                elif isinstance(ticket, int):
                    return ticket
                elif isinstance(ticket, str) and ticket.isdigit():
                    return int(ticket)
            except (ValueError, TypeError):
                continue
    
    return 0

def generate_symbol_impact_matrix(symbol_journeys: Dict[str, List[Dict]]) -> Dict:
    """Generate current symbol impact matrix by agent status."""
    
    # Simplified matrix for visualization
    matrix = {
        'vote_symbols': {},
        'bind_symbols': {},
        'civil_symbols': {},
        'agent_status_effects': {
            'regular': {'vote_multiplier': 1.0, 'bind_access': False},
            'binder': {'vote_multiplier': 1.5, 'bind_access': True},
            'demoted': {'vote_multiplier': 0.2, 'bind_access': False}
        }
    }
    
    # Process recent interpretations to build current state
    for symbol, journey in symbol_journeys.items():
        if not journey:
            continue
            
        latest_event = journey[-1]  # Most recent interpretation
        interpretation = latest_event.get('interpretation', {})
        
        if symbol.startswith(('promote_', 'demote_')):
            matrix['vote_symbols'][symbol] = {
                'latest_multiplier': interpretation.get('reputation_multiplier', 1.0),
                'semantic_evolution': bool(interpretation.get('semantic_change')),
                'tickets_active': len(journey)
            }
        elif 'BIND' in symbol.upper():
            matrix['bind_symbols'][symbol] = {
                'access_restricted': interpretation.get('action') == 'BLOCKED',
                'tickets_active': len(journey)
            }
        elif 'CIVIL' in symbol.upper():
            matrix['civil_symbols'][symbol] = {
                'toxicity_detected': interpretation.get('action') == 'BLOCKED',
                'tickets_active': len(journey)
            }
    
    return matrix

if __name__ == "__main__":
    sys.exit(main())