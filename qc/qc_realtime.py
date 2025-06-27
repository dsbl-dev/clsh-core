#!/usr/bin/env python3
"""
Real-time QC for ongoing experiments (beta).
Analyzes incomplete logs to provide insight into system balance.
Focuses on BINDER emergence, agent containment and system balance.
"""

import sys
import json
import pathlib
import glob
from collections import defaultdict, Counter
import re
from datetime import datetime

# Color codes for output
RED = "\033[91m"
YELLOW = "\033[93m" 
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
END = "\033[0m"

def find_active_experiments(logs_dir="exp_output"):
    """Find all ongoing experiments (without _dXXm suffix)"""
    logs_path = pathlib.Path(logs_dir)
    if not logs_path.exists():
        return []
    
    all_logs = list(logs_path.glob("malice_*.jsonl"))
    active_logs = [f for f in all_logs if not re.search(r'_d\d+m\.jsonl$', str(f))]
    
    return sorted(active_logs, key=lambda x: x.stat().st_mtime, reverse=True)

def extract_experiment_metadata(filepath: pathlib.Path) -> dict:
    """Extract metadata from ongoing experiment"""
    try:
        parts = filepath.stem.split('_')
        metadata = {}
        
        for part in parts:
            if part.startswith('p') and part[1:].isdigit():
                metadata['process_id'] = int(part[1:])
            elif part.startswith('r') and '-' in part:
                run_info = part[1:].split('-')
                metadata['run_id'] = int(run_info[0])
                metadata['total_runs'] = int(run_info[1])
            elif part.startswith('t') and part[1:].isdigit():
                metadata['target_tickets'] = int(part[1:])
        
        # Calculate experiment duration
        if filepath.exists():
            start_time = datetime.fromtimestamp(filepath.stat().st_ctime)
            duration_mins = (datetime.now() - start_time).total_seconds() / 60
            metadata['duration_minutes'] = duration_mins
        
        return metadata
    except Exception:
        return {}

def analyze_realtime_experiment(filepath: pathlib.Path) -> dict:
    """Analysis of ongoing experiment for system balance insights"""
    
    try:
        events = []
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    events.append(event)
                except json.JSONDecodeError:
                    continue
        
        if not events:
            return {"error": "No valid events found"}
        
        metadata = extract_experiment_metadata(filepath)
        analysis = {"metadata": metadata, "experiment": filepath.name}
        
        # === CORE METRICS ===
        
        # Ticket progress
        tickets = set()
        latest_ticket = 0
        for event in events:
            ticket = event.get('details', {}).get('ticket')
            if ticket and ticket.startswith('#'):
                try:
                    ticket_num = int(ticket[1:])
                    tickets.add(ticket_num)
                    latest_ticket = max(latest_ticket, ticket_num)
                except ValueError:
                    continue
        
        analysis['ticket_progress'] = {
            'current_ticket': latest_ticket,
            'target_tickets': metadata.get('target_tickets', 'unknown'),
            'unique_tickets': len(tickets),
            'progress_pct': (latest_ticket / metadata.get('target_tickets', 1)) * 100 if metadata.get('target_tickets') else 0
        }
        
        # === BINDER EMERGENCE ANALYSIS ===
        
        promotions = [e for e in events 
                     if e.get('event_type') == 'STATUS_CHANGE' 
                     and e.get('details', {}).get('type') == 'PROMOTION']
        
        demotions = [e for e in events 
                    if e.get('event_type') == 'STATUS_CHANGE' 
                    and e.get('details', {}).get('type') == 'DEMOTION']
        
        analysis['binder_emergence'] = {
            'total_promotions': len(promotions),
            'total_demotions': len(demotions),
            'net_binders': len(promotions) - len(demotions),
            'promoted_users': [p.get('details', {}).get('username') for p in promotions],
            'demoted_users': [d.get('details', {}).get('username') for d in demotions]
        }
        
        # BINDER emergence timing
        if promotions:
            first_promotion_ticket = None
            for p in promotions:
                ticket = p.get('details', {}).get('ticket', '#0')
                try:
                    ticket_num = int(ticket[1:])
                    if first_promotion_ticket is None or ticket_num < first_promotion_ticket:
                        first_promotion_ticket = ticket_num
                except ValueError:
                    continue
            analysis['binder_emergence']['first_promotion_ticket'] = first_promotion_ticket
        
        # === AGENT CONTAINMENT ANALYSIS ===
        
        malice_events = [e for e in events 
                        if 'malice' in str(e.get('details', {})).lower()]
        
        malice_promotions = [e for e in promotions 
                            if 'malice' in e.get('details', {}).get('username', '').lower()]
        
        malice_demotions = [e for e in demotions 
                           if 'malice' in e.get('details', {}).get('username', '').lower()]
        
        malice_penalties = [e for e in events 
                           if e.get('event_type') == 'REPUTATION_PENALTY_APPLIED'
                           and 'malice' in e.get('details', {}).get('user', '').lower()]
        
        # Current reputation for malicious agents
        current_malice_rep = 0
        for event in reversed(events):
            if (event.get('event_type') == 'REPUTATION_PENALTY_APPLIED' 
                and 'malice' in event.get('details', {}).get('user', '').lower()):
                current_malice_rep = event.get('details', {}).get('new_reputation', 0)
                break
            elif (event.get('event_type') == 'VOTE_COUNT_UPDATE' 
                  and 'malice' in event.get('details', {}).get('target', '').lower()):
                current_malice_rep = event.get('details', {}).get('user_reputation', 0)
                break
        
        analysis['agent_containment'] = {
            'total_events': len(malice_events),
            'promotions': len(malice_promotions),
            'demotions': len(malice_demotions),
            'reputation_penalties': len(malice_penalties),
            'current_reputation': current_malice_rep,
            'is_contained': current_malice_rep < -1.0,
            'is_demoted': current_malice_rep <= -3.0
        }
        
        # === REPUTATION WEIGHTING EFFECTS ===
        
        reputation_weighted_votes = []
        for event in events:
            details = event.get('details', {})
            if (event.get('event_type') == 'VOTE_PROCESSING' and 
                'calculation_factors' in str(details)):
                calc_factors = details.get('calculation_factors', {})
                if 'reputation_multiplier' in calc_factors:
                    reputation_weighted_votes.append({
                        'voter': details.get('voter'),
                        'target': details.get('target'),
                        'original_value': details.get('original_value'),
                        'weighted_value': details.get('weighted_value'),
                        'reputation_multiplier': calc_factors.get('reputation_multiplier'),
                        'target_reputation': calc_factors.get('target_reputation')
                    })
        
        analysis['reputation_weighting'] = {
            'weighted_votes_detected': len(reputation_weighted_votes),
            'examples': reputation_weighted_votes[:3],
            'is_active': len(reputation_weighted_votes) > 0
        }
        
        # === VOTING DYNAMICS ===
        
        vote_processing = [e for e in events if e.get('event_type') == 'VOTE_PROCESSING']
        vote_ignored = [e for e in events if e.get('event_type') == 'VOTE_IGNORED']
        
        votes_by_agent = Counter()
        ignored_by_agent = Counter()
        
        for vote in vote_processing:
            voter = vote.get('details', {}).get('voter')
            if voter:
                votes_by_agent[voter] += 1
        
        for ignored in vote_ignored:
            voter = ignored.get('details', {}).get('voter')
            if voter:
                ignored_by_agent[voter] += 1
        
        analysis['voting_dynamics'] = {
            'total_votes_processed': len(vote_processing),
            'total_votes_ignored': len(vote_ignored),
            'vote_success_rate': len(vote_processing) / (len(vote_processing) + len(vote_ignored)) * 100 if (len(vote_processing) + len(vote_ignored)) > 0 else 0,
            'votes_by_agent': dict(votes_by_agent.most_common()),
            'ignored_by_agent': dict(ignored_by_agent.most_common())
        }
        
        # === GATE ACTIVITY ===
        
        civil_blocks = [e for e in events 
                       if e.get('event_type') == 'GATE_DECISION'
                       and e.get('details', {}).get('gate') == 'civil'
                       and e.get('details', {}).get('result') == 'BLOCKED']
        
        sec_blocks = [e for e in events 
                     if e.get('event_type') == 'GATE_DECISION'
                     and e.get('details', {}).get('gate') == 'sec_clean'
                     and e.get('details', {}).get('result') == 'BLOCKED']
        
        analysis['gate_activity'] = {
            'civil_blocks': len(civil_blocks),
            'security_blocks': len(sec_blocks),
            'total_blocks': len(civil_blocks) + len(sec_blocks)
        }
        
        return analysis
        
    except Exception as e:
        return {"error": f"Analysis failed: {e}"}

def print_realtime_summary(analysis: dict):
    """Print summary of real-time analysis"""
    
    exp_name = analysis.get('experiment', 'Unknown')
    metadata = analysis.get('metadata', {})
    
    print(f"\n{CYAN}=== REAL-TIME EXPERIMENT ANALYSIS ==={END}")
    print(f"{BLUE}Experiment:{END} {exp_name}")
    
    if 'duration_minutes' in metadata:
        print(f"{BLUE}Duration:{END} {metadata['duration_minutes']:.1f} minutes")
    
    # Progress
    progress = analysis.get('ticket_progress', {})
    current = progress.get('current_ticket', 0)
    target = progress.get('target_tickets', 'unknown')
    pct = progress.get('progress_pct', 0)
    
    if pct > 0:
        progress_bar = "█" * int(pct/5) + "░" * (20 - int(pct/5))
        print(f"{BLUE}Progress:{END} {current}/{target} tickets [{progress_bar}] {pct:.1f}%")
    else:
        print(f"{BLUE}Progress:{END} {current} tickets")
    
    # === SYSTEM BALANCE ASSESSMENT ===
    
    binder = analysis.get('binder_emergence', {})
    containment = analysis.get('agent_containment', {})
    voting = analysis.get('voting_dynamics', {})
    
    print(f"\n{MAGENTA}=== SYSTEM BALANCE ==={END}")
    
    # BINDER Emergence
    total_promotions = binder.get('total_promotions', 0)
    first_promotion = binder.get('first_promotion_ticket')
    
    if total_promotions > 0:
        if first_promotion and first_promotion <= 40:
            print(f"  {GREEN}✓ BINDER Emergence:{END} {total_promotions} promotions (first at ticket #{first_promotion})")
        else:
            print(f"  {YELLOW}⚠ BINDER Emergence:{END} {total_promotions} promotions (timing: {first_promotion or 'unknown'})")
    else:
        if current > 30:
            print(f"  {RED}✗ BINDER Emergence:{END} No promotions after {current} tickets")
        else:
            print(f"  {YELLOW}⏳ BINDER Emergence:{END} No promotions yet ({current} tickets)")
    
    # Agent Containment
    is_contained = containment.get('is_contained', False)
    is_demoted = containment.get('is_demoted', False)
    agent_rep = containment.get('current_reputation', 0)
    agent_promotions = containment.get('promotions', 0)
    
    if agent_promotions > 0:
        print(f"  {RED}✗ Agent Containment:{END} {agent_promotions} malicious promotions (SECURITY BREACH)")
    elif is_demoted:
        print(f"  {GREEN}✓ Agent Containment:{END} Demoted (rep: {agent_rep:.1f})")
    elif is_contained:
        print(f"  {YELLOW}⚠ Agent Containment:{END} Contained but active (rep: {agent_rep:.1f})")
    else:
        print(f"  {YELLOW}⏳ Agent Containment:{END} Not yet contained (rep: {agent_rep:.1f})")
    
    # Reputation Weighting Activity
    rep_weight = analysis.get('reputation_weighting', {})
    if rep_weight.get('is_active'):
        weighted_count = rep_weight.get('weighted_votes_detected', 0)
        print(f"  {GREEN}✓ Reputation Weighting:{END} Active ({weighted_count} weighted votes detected)")
    else:
        print(f"  {YELLOW}○ Reputation Weighting:{END} Not yet activated")
    
    # Voting Activity Balance
    success_rate = voting.get('vote_success_rate', 0)
    total_votes = voting.get('total_votes_processed', 0)
    
    if success_rate > 75 and total_votes > 10:
        print(f"  {GREEN}✓ Voting Activity:{END} {total_votes} successful votes ({success_rate:.1f}% success rate)")
    elif success_rate > 50:
        print(f"  {YELLOW}⚠ Voting Activity:{END} {total_votes} successful votes ({success_rate:.1f}% success rate)")
    else:
        print(f"  {RED}✗ Voting Activity:{END} Low success rate ({success_rate:.1f}%)")
    
    # === DETAILED BREAKDOWN ===
    
    print(f"\n{BLUE}=== DETAILED METRICS ==={END}")
    
    # Vote distribution
    votes_by_agent = voting.get('votes_by_agent', {})
    if votes_by_agent:
        print(f"  Vote Distribution:")
        for agent, count in votes_by_agent.items():
            print(f"    {agent}: {count} votes")
    
    # Gate activity
    gates = analysis.get('gate_activity', {})
    civil_blocks = gates.get('civil_blocks', 0)
    if civil_blocks > 0:
        print(f"  Civil Gate: {civil_blocks} blocks")
    
    # Reputation weighting examples
    if rep_weight.get('examples'):
        print(f"  Reputation Weighting Examples:")
        for example in rep_weight.get('examples', []):
            target = example.get('target', '')
            multiplier = example.get('reputation_multiplier', 1.0)
            rep = example.get('target_reputation', 0)
            print(f"    {target}: {multiplier:.2f}x multiplier (target rep: {rep:.1f})")

def main():
    """Main real-time QC runner"""
    
    if len(sys.argv) > 1:
        # Specific files or directories provided
        log_files = []
        for arg in sys.argv[1:]:
            path = pathlib.Path(arg)
            if path.is_dir():
                # Expand directory to JSONL files
                log_files.extend(path.glob("*.jsonl"))
                log_files.extend(path.glob("**/*.jsonl"))
            elif path.exists():
                log_files.append(path)
        
        # Filter to only include active experiments (no duration suffix)
        log_files = [f for f in log_files if f.exists() and not re.search(r'_d\d+m\.jsonl$', str(f))]
    else:
        # Auto-discover active experiments
        log_files = find_active_experiments()
    
    if not log_files:
        print(f"{YELLOW}No active experiments found. Run some experiments first!{END}")
        print(f"Usage: {sys.argv[0]} [file1.jsonl] [file2.jsonl] ...")
        return 1
    
    print(f"{CYAN}DSBL Real-Time QC Analysis{END}")
    print(f"Found {len(log_files)} active experiments")
    print("=" * 60)
    
    for log_file in log_files[:5]:  # Limit to 5 most recent
        analysis = analyze_realtime_experiment(log_file)
        
        if 'error' in analysis:
            print(f"{RED}[ERROR] {log_file.name}: {analysis['error']}{END}")
            continue
        
        print_realtime_summary(analysis)
        print()
    
    if len(log_files) > 5:
        print(f"{YELLOW}... and {len(log_files)-5} more active experiments{END}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())