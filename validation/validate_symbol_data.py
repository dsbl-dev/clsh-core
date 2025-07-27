#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 DSBL-Dev contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Symbol Journey Data Validation Script
Validates Symbol Journey Timeline tracking captures the right events
"""

import json
import sys
import pathlib
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

# Colors for output
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
END = "\033[0m"

def load_experiment_log(log_file: pathlib.Path) -> List[Dict]:
    """Load JSONL experiment log file."""
    events = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                event = json.loads(line.strip())
                events.append(event)
            except json.JSONDecodeError as e:
                print(f"{YELLOW}Warning: JSON error on line {line_num}: {e}{END}")
                continue
    return events

def validate_symbol_journey_data(log_files: List[pathlib.Path]) -> Dict:
    """Validate Symbol Journey Timeline data quality."""
    
    validation_results = {
        "files_processed": len(log_files),
        "total_events": 0,
        "symbol_interpretations": 0,
        "status_changes": 0,
        "vote_multiplier_effects": 0,
        "reputation_weighting_effects": 0,
        "immune_frequency_adjustments": 0,
        "adaptive_adjustments": 0,
        "unique_symbol_journeys": 0,
        "data_quality_score": 0.0,
        "issues": []
    }
    
    all_symbol_interpretations = []
    all_status_changes = []
    symbol_journeys = defaultdict(list)
    
    for log_file in log_files:
        print(f"📁 Processing: {log_file.name}")
        
        try:
            events = load_experiment_log(log_file)
            validation_results["total_events"] += len(events)
            
            # Extract specific event types
            symbol_interpretations = [e for e in events if e.get('event_type') == 'SYMBOL_INTERPRETATION']
            status_changes = [e for e in events if e.get('event_type') == 'STATUS_CHANGE']
            vote_processing = [e for e in events if e.get('event_type') == 'VOTE_PROCESSING']
            
            validation_results["symbol_interpretations"] += len(symbol_interpretations)
            validation_results["status_changes"] += len(status_changes)
            
            all_symbol_interpretations.extend(symbol_interpretations)
            all_status_changes.extend(status_changes)
            
            # Check for BINDER vote multiplier effects
            binder_votes = [v for v in vote_processing 
                           if v.get('details', {}).get('is_binder_vote') == True]
            validation_results["vote_multiplier_effects"] += len(binder_votes)
            
            # Check for reputation weighting effects  
            reputation_weighted = [e for e in events 
                                 if e.get('event_type') == 'SYMBOL_INTERPRETATION' 
                                 and e.get('details', {}).get('symbol_type') == 'VOTE_WEIGHTING']
            validation_results["reputation_weighting_effects"] += len(reputation_weighted)
            
            # Build symbol journeys
            for interpretation in symbol_interpretations:
                details = interpretation.get('details', {})
                symbol_content = details.get('symbol_content')
                if symbol_content:
                    symbol_journeys[symbol_content].append(interpretation)
                    
        except Exception as e:
            validation_results["issues"].append(f"Error processing {log_file.name}: {e}")
    
    validation_results["unique_symbol_journeys"] = len(symbol_journeys)
    
    # Validate data quality
    quality_checks = []
    
    # 1. Symbol interpretation coverage
    if validation_results["symbol_interpretations"] > 0:
        quality_checks.append(("Symbol interpretations logged", True))
    else:
        quality_checks.append(("Symbol interpretations logged", False))
        validation_results["issues"].append("No SYMBOL_INTERPRETATION events found")
    
    # 2. Status change tracking
    if validation_results["status_changes"] > 0:
        quality_checks.append(("Status changes tracked", True))
        
        # Check for new time-tracking fields
        has_promotion_ticket = any(
            'promotion_ticket' in sc.get('details', {}) 
            for sc in all_status_changes 
            if sc.get('details', {}).get('type') == 'PROMOTION'
        )
        quality_checks.append(("Time-tracking fields present", has_promotion_ticket))
        
        if not has_promotion_ticket:
            validation_results["issues"].append("Missing promotion_ticket in STATUS_CHANGE events")
    else:
        quality_checks.append(("Status changes tracked", False))
        validation_results["issues"].append("No STATUS_CHANGE events found")
    
    # 3. BINDER vote multiplier detection
    if validation_results["vote_multiplier_effects"] > 0:
        quality_checks.append(("BINDER vote multipliers detected", True))
    else:
        quality_checks.append(("BINDER vote multipliers detected", False))
        # Check if we have BINDER promotions but no BINDER votes
        if validation_results["status_changes"] > 0:
            promotion_events = [sc for sc in all_status_changes 
                              if sc.get('details', {}).get('type') == 'PROMOTION']
            if promotion_events:
                validation_results["issues"].append("BINDER promotions found but no BINDER votes - experiments may be too short")
            else:
                validation_results["issues"].append("No BINDER vote multiplier effects found")
        else:
            validation_results["issues"].append("No BINDER vote multiplier effects found")
    
    # 4. Symbol journey diversity
    if validation_results["unique_symbol_journeys"] >= 3:
        quality_checks.append(("Diverse symbol journeys", True))
    else:
        quality_checks.append(("Diverse symbol journeys", False))
        validation_results["issues"].append(f"Only {validation_results['unique_symbol_journeys']} unique symbol journeys")
    
    # 5. Check for STATUS_CHANGE symbol interpretations
    status_symbol_interpretations = [
        si for si in all_symbol_interpretations 
        if si.get('details', {}).get('symbol_type') == 'STATUS_CHANGE'
    ]
    if status_symbol_interpretations:
        quality_checks.append(("Status change symbol tracking", True))
    else:
        quality_checks.append(("Status change symbol tracking", False))
        validation_results["issues"].append("No STATUS_CHANGE symbol interpretations found")
    
    # Calculate quality score
    passed_checks = sum(1 for _, passed in quality_checks if passed)
    validation_results["data_quality_score"] = (passed_checks / len(quality_checks)) * 100
    
    # Detailed analysis
    validation_results["symbol_journey_analysis"] = analyze_symbol_journeys(symbol_journeys)
    validation_results["quality_checks"] = quality_checks
    
    return validation_results

def analyze_symbol_journeys(symbol_journeys: Dict[str, List]) -> Dict:
    """Analyze symbol journey patterns."""
    
    analysis = {
        "most_tracked_symbols": [],
        "symbols_with_context_changes": [],
        "average_interpretations_per_symbol": 0.0
    }
    
    # Most tracked symbols
    symbol_counts = {symbol: len(journey) for symbol, journey in symbol_journeys.items()}
    sorted_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)
    analysis["most_tracked_symbols"] = sorted_symbols[:5]
    
    # Average interpretations
    if symbol_journeys:
        total_interpretations = sum(len(journey) for journey in symbol_journeys.values())
        analysis["average_interpretations_per_symbol"] = total_interpretations / len(symbol_journeys)
    
    # Symbols with context changes (same symbol, different effects)
    for symbol, journey in symbol_journeys.items():
        if len(journey) >= 2:
            # Check for different interpretation effects
            effects = set()
            for event in journey:
                interpretation = event.get('details', {}).get('interpretation', {})
                action = interpretation.get('action', 'unknown')
                effects.add(action)
            
            if len(effects) > 1:
                analysis["symbols_with_context_changes"].append({
                    "symbol": symbol,
                    "interpretation_count": len(journey),
                    "different_effects": list(effects)
                })
    
    return analysis

def print_validation_report(results: Dict):
    """Print validation results in a readable format."""
    
    print(f"\n{BLUE}=== SYMBOL JOURNEY DATA VALIDATION REPORT ==={END}")
    print(f"Files processed: {results['files_processed']}")
    print(f"Total events: {results['total_events']}")
    print("")
    
    # Quality score
    score = results['data_quality_score']
    if score >= 80:
        score_color = GREEN
        score_status = "EXCELLENT"
    elif score >= 60:
        score_color = YELLOW
        score_status = "GOOD"
    else:
        score_color = RED
        score_status = "NEEDS IMPROVEMENT"
    
    print(f"Data Quality Score: {score_color}{score:.1f}% ({score_status}){END}")
    print("")
    
    # Key metrics
    print(f"{BLUE}=== KEY METRICS ==={END}")
    print(f"✓ Symbol interpretations: {results['symbol_interpretations']}")
    print(f"✓ Status changes: {results['status_changes']}")
    print(f"✓ BINDER vote multipliers: {results['vote_multiplier_effects']}")
    print(f"✓ Reputation weighting: {results['reputation_weighting_effects']}")
    print(f"✓ Unique symbol journeys: {results['unique_symbol_journeys']}")
    print("")
    
    # Quality checks
    print(f"{BLUE}=== QUALITY CHECKS ==={END}")
    for check_name, passed in results['quality_checks']:
        status = f"{GREEN}✓{END}" if passed else f"{RED}✗{END}"
        print(f"{status} {check_name}")
    print("")
    
    # Symbol journey analysis
    journey_analysis = results.get('symbol_journey_analysis', {})
    if journey_analysis.get('most_tracked_symbols'):
        print(f"{BLUE}=== TOP TRACKED SYMBOLS ==={END}")
        for symbol, count in journey_analysis['most_tracked_symbols']:
            print(f"  {symbol}: {count} interpretations")
        print("")
    
    # Context changes
    context_changes = journey_analysis.get('symbols_with_context_changes', [])
    if context_changes:
        print(f"{BLUE}=== SYMBOLS WITH CONTEXT-DEPENDENT CHANGES ==={END}")
        for change in context_changes[:3]:  # Show top 3
            symbol = change['symbol']
            count = change['interpretation_count']
            effects = ', '.join(change['different_effects'])
            print(f"  {symbol}: {count} interpretations, effects: {effects}")
        print("")
    
    # Issues
    if results['issues']:
        print(f"{YELLOW}=== ISSUES FOUND ==={END}")
        for issue in results['issues']:
            print(f"  {issue}")
        print("")
    
    # Recommendations
    print(f"{BLUE}=== RECOMMENDATIONS ==={END}")
    
    if results['symbol_interpretations'] < 10:
        print("🔧 Run longer experiments to capture more symbol interpretations")
    
    if results['vote_multiplier_effects'] == 0:
        print("🔧 Ensure experiments run long enough for BINDER promotions")
    
    if results['unique_symbol_journeys'] < 5:
        print("🔧 Increase diversity by running multiple experiments")
    
    if score >= 80:
        print(f"{GREEN}Data quality is excellent! Ready for large-scale data collection.{END}")
    elif score >= 60:
        print(f"{YELLOW}Data quality is good. Consider addressing minor issues before scaling.{END}")
    else:
        print(f"{RED}Data quality needs improvement. Fix issues before proceeding.{END}")

def main():
    """Main validation runner."""
    
    if len(sys.argv) > 1:
        # Specific files provided
        log_files = [pathlib.Path(p) for p in sys.argv[1:]]
        log_files = [f for f in log_files if f.exists()]
    else:
        # Auto-discover recent log files
        logs_dir = pathlib.Path("logs")
        if logs_dir.exists():
            log_files = list(logs_dir.glob("malice_*.jsonl"))
            # Take the 3 most recent
            log_files = sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]
        else:
            log_files = []
    
    if not log_files:
        print(f"{RED}No log files found. Run some experiments first!{END}")
        print(f"Usage: {sys.argv[0]} [file1.jsonl] [file2.jsonl] ...")
        return 1
    
    print(f"{BLUE}Symbol Journey Data Validator{END}")
    print(f"Validating {len(log_files)} log files...")
    print("=" * 50)
    
    # Run validation
    results = validate_symbol_journey_data(log_files)
    
    # Print report
    print_validation_report(results)
    
    # Return status code based on quality
    if results['data_quality_score'] >= 60:
        return 0  # Success
    else:
        return 1  # Issues found

if __name__ == "__main__":
    sys.exit(main())