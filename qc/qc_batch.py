#!/usr/bin/env python3
"""
QC-skript för DSBL-batchar.
Kontrollerar dataintegritet och experiment-validitet för JSONL audit logs.
Kräver endast standard Python-bibliotek.
"""

import sys
import json
import pathlib
from collections import defaultdict
import re

# Färgkoder för output
RED = "\033[91m"
YELLOW = "\033[93m" 
GREEN = "\033[92m"
BLUE = "\033[94m"
END = "\033[0m"

def extract_run_metadata(filepath: pathlib.Path) -> dict:
    """Extrahera metadata från filnamn: malice_YYMMDD_HHhMMmSSs_pPID_rXX_tZZ_dMM"""
    try:
        parts = filepath.stem.split('_')
        metadata = {}
        
        for part in parts:
            if part.startswith('p') and part[1:].isdigit():
                metadata['process_id'] = int(part[1:])
            elif part.startswith('r') and part[1:].isdigit():
                metadata['run_id'] = int(part[1:])
            elif part.startswith('t') and part[1:].isdigit():
                metadata['expected_tickets'] = int(part[1:])
            elif part.startswith('d') and part[1:-1].isdigit() and part.endswith('m'):
                metadata['duration_minutes'] = int(part[1:-1])
        
        return metadata
    except Exception as e:
        print(f"{YELLOW}[WARN] Could not parse metadata from {filepath.name}: {e}{END}")
        return {}

def qc_file(filepath: pathlib.Path) -> bool:
    """Return True if file passes all hard checks."""
    print(f"\n{BLUE}Checking: {filepath.name}{END}")
    
    hard_fail = False
    warn = False
    
    try:
        # Läs JSONL fil rad för rad för bättre felhantering
        events = []
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    event = json.loads(line.strip())
                    events.append(event)
                except json.JSONDecodeError as e:
                    print(f"{RED}[FAIL] {filepath.name}: JSON error on line {line_num}: {e}{END}")
                    hard_fail = True
        
        if not events:
            print(f"{RED}[FAIL] {filepath.name}: Empty file{END}")
            return False
            
        # Använd events-listan direkt istället för pandas DataFrame
        metadata = extract_run_metadata(filepath)
        
        # A. Ticket count validation - check for reasonable coverage, not exact match
        if 'expected_tickets' in metadata:
            expected = metadata['expected_tickets']
            tickets_found = set()
            ticket_numbers = []
            
            for event in events:
                details = event.get('details', {})
                ticket = details.get('ticket')
                if ticket and ticket.startswith('#'):
                    tickets_found.add(ticket)
                    try:
                        ticket_numbers.append(int(ticket[1:]))
                    except ValueError:
                        continue
            
            actual_tickets = len(tickets_found)
            
            # NEW: Handle parallel batch runner ticket numbering
            if ticket_numbers:
                min_ticket = min(ticket_numbers)
                max_ticket = max(ticket_numbers)
                
                # For parallel runs, tickets are offset by run_id * 1000
                # e.g., run 1: #1001-#1040, run 2: #2001-#2040, etc.
                if 'run_id' in metadata:
                    run_id = metadata['run_id']
                    expected_start = run_id * 1000 + 1
                    expected_end = run_id * 1000 + expected
                    
                    # Check if tickets are in expected range for this run
                    in_range_count = sum(1 for t in ticket_numbers 
                                       if expected_start <= t <= expected_end)
                    coverage = in_range_count / expected if expected > 0 else 0
                    
                    if coverage < 0.6:  # At least 60% of expected tickets for this run
                        print(f"{RED}[FAIL] {filepath.name}: Poor ticket coverage {in_range_count}/{expected} tickets ({coverage:.1%}){END}")
                        hard_fail = True
                    elif coverage < 0.8:  # At least 80% for warning
                        print(f"{YELLOW}[WARN] {filepath.name}: {in_range_count} active tickets (expected ~{expected}, coverage: {coverage:.1%}){END}")
                        warn = True
                    else:
                        print(f"  ✓ {in_range_count} active tickets out of {expected} expected ({coverage:.1%} coverage)")
                else:
                    # Fallback to old logic for non-parallel runs
                    ticket_range = max_ticket - min_ticket + 1
                    coverage = actual_tickets / ticket_range if ticket_range > 0 else 0
                    
                    if coverage < 0.6:
                        print(f"{RED}[FAIL] {filepath.name}: Poor ticket coverage {actual_tickets}/{ticket_range} tickets ({coverage:.1%}){END}")
                        hard_fail = True
                    elif actual_tickets < expected * 0.7:
                        print(f"{YELLOW}[WARN] {filepath.name}: {actual_tickets} active tickets (expected ~{expected}, range coverage: {coverage:.1%}){END}")
                        warn = True
                    else:
                        print(f"  ✓ {actual_tickets} active tickets out of {ticket_range} range ({coverage:.1%} coverage)")
            else:
                print(f"{RED}[FAIL] {filepath.name}: No valid tickets found{END}")
                hard_fail = True
        
        # B. Unique & monotonic message_id validation
        message_ids = []
        for event in events:
            if 'details' in event and isinstance(event['details'], dict):
                if 'message_id' in event['details']:
                    message_ids.append(event['details']['message_id'])
        
        if message_ids:
            if len(message_ids) != len(set(message_ids)):
                print(f"{RED}[FAIL] {filepath.name}: Duplicate message_id detected{END}")
                hard_fail = True
            elif message_ids != sorted(message_ids):
                print(f"{RED}[FAIL] {filepath.name}: Non-monotonic message_id sequence{END}")
                hard_fail = True
        
        # C. Ticket chronology
        tickets = []
        for event in events:
            if 'details' in event and isinstance(event['details'], dict):
                ticket = event['details'].get('ticket')
                if ticket and ticket.startswith('#'):
                    try:
                        tickets.append(int(ticket[1:]))
                    except ValueError:
                        continue
        
        if tickets:
            # For parallel runs, expect many duplicate ticket numbers (normal behavior)
            # Only warn if duplicates are excessive
            unique_tickets = len(set(tickets))
            total_tickets = len(tickets)
            duplicates = total_tickets - unique_tickets
            
            if 'run_id' in metadata:
                # Parallel run: expect some duplicates but not excessive
                duplicate_ratio = duplicates / total_tickets if total_tickets > 0 else 0
                if duplicate_ratio > 0.8:  # More than 80% duplicates is suspicious
                    print(f"{YELLOW}[WARN] {filepath.name}: High duplicate ratio {duplicates}/{total_tickets} ({duplicate_ratio:.1%}){END}")
                    warn = True
                else:
                    # Just informational for parallel runs
                    print(f"  ✓ {unique_tickets} unique tickets ({duplicates} duplicates expected in parallel run)")
            else:
                # Single run: duplicates are problematic
                if duplicates > 0:
                    print(f"{YELLOW}[WARN] {filepath.name}: {duplicates} duplicate ticket numbers{END}")
                    warn = True
            
            # Kontrollera att tickets ökar generellt (men inte nödvändigtvis strikt monotont)
            if tickets != sorted(tickets):
                # Räkna hur många som är "out of order"
                out_of_order = sum(1 for i in range(1, len(tickets)) if tickets[i] < tickets[i-1])
                if out_of_order > len(tickets) * 0.1:  # Mer än 10% out of order
                    print(f"{RED}[FAIL] {filepath.name}: Severely non-chronological tickets ({out_of_order} reversals){END}")
                    hard_fail = True
        
        # D. Status consistency - demoted users shouldn't vote
        demoted_users = set()
        vote_violations = []
        
        for event in events:
            if event.get('event_type') == 'STATUS_CHANGE':
                details = event.get('details', {})
                if details.get('to_status') == 'demoted':
                    demoted_users.add(details.get('username'))
            
            elif event.get('event_type') == 'VOTE_PROCESSING':
                details = event.get('details', {})
                voter = details.get('voter')
                if voter in demoted_users:
                    vote_violations.append(voter)
        
        if vote_violations:
            print(f"{RED}[FAIL] {filepath.name}: {len(vote_violations)} illegal votes from demoted users{END}")
            hard_fail = True
        
        # E. BINDER emergence check
        binder_promotions = [e for e in events 
                           if e.get('event_type') == 'STATUS_CHANGE' 
                           and e.get('details', {}).get('type') == 'PROMOTION']
        
        if not binder_promotions:
            print(f"{YELLOW}[WARN] {filepath.name}: No BINDER promotions found (experiment might need longer duration){END}")
            warn = True
        else:
            print(f"  ✓ Found {len(binder_promotions)} BINDER promotion(s)")
        
        # F. Gate consistency - ALLOWED gates shouldn't immediately cause penalties
        # BUT: sec_clean ALLOWED → civil penalty is normal (different gates)
        gate_inconsistencies = 0
        for i, event in enumerate(events[:-1]):
            if (event.get('event_type') == 'GATE_DECISION' and 
                event.get('details', {}).get('result') == 'ALLOWED'):
                
                next_event = events[i + 1]
                if next_event.get('event_type') == 'REPUTATION_PENALTY_APPLIED':
                    # Only flag if same gate causes ALLOWED → penalty
                    current_gate = event.get('details', {}).get('gate')
                    # Penalty events don't have gate field, so we check if it's likely the same gate
                    # If current gate is sec_clean, next penalty is likely from civil gate (OK)
                    if current_gate != 'sec_clean':  # sec_clean → civil penalty is normal
                        gate_inconsistencies += 1
        
        if gate_inconsistencies > 0:
            print(f"{RED}[FAIL] {filepath.name}: {gate_inconsistencies} gate inconsistencies (ALLOWED→penalty){END}")
            hard_fail = True
        
        # G. Grundläggande statistik
        civil_blocks = len([e for e in events 
                          if e.get('event_type') == 'GATE_DECISION' 
                          and e.get('details', {}).get('gate') == 'civil'
                          and e.get('details', {}).get('result') == 'BLOCKED'])
        
        mallory_penalties = len([e for e in events 
                               if e.get('event_type') == 'REPUTATION_PENALTY_APPLIED'
                               and e.get('details', {}).get('user') == 'mallory'])
        
        print(f"  ✓ {civil_blocks} CIVIL gate blocks")
        print(f"  ✓ {mallory_penalties} Mallory reputation penalties")
        
        # H. Mallory post-to-vote ratio analysis (v2.9)
        mallory_posts = 0
        mallory_votes = 0
        
        for event in events:
            details = event.get('details', {})
            
            # Count posts from Mallory (MESSAGE_SENT events where sender is mallory)
            if (event.get('event_type') == 'MESSAGE_SENT' and 
                details.get('sender') == 'mallory'):
                mallory_posts += 1
            
            # Count successful votes from Mallory
            elif (event.get('event_type') == 'VOTE_PROCESSING' and 
                  details.get('voter') == 'mallory'):
                mallory_votes += 1
        
        if mallory_posts > 0:
            post_vote_ratio = mallory_posts / mallory_votes if mallory_votes > 0 else float('inf')
            if post_vote_ratio >= 2.0:  # At least 2 posts per vote = active antagonist
                print(f"  ✓ Mallory post-to-vote ratio: {post_vote_ratio:.1f} (active antagonist)")
            elif post_vote_ratio >= 1.0:
                print(f"{YELLOW}[WARN] Mallory post-to-vote ratio: {post_vote_ratio:.1f} (moderate activity){END}")
                warn = True
            else:
                print(f"{YELLOW}[WARN] Mallory post-to-vote ratio: {post_vote_ratio:.1f} (vote-heavy behavior){END}")
                warn = True
        else:
            print(f"{YELLOW}[WARN] No Mallory posts detected (ghost behavior){END}")
            warn = True
        
        # Slutlig bedömning
        if not hard_fail:
            status_color = GREEN if not warn else YELLOW
            status_text = "PASS" if not warn else "PASS (with warnings)"
            print(f"{status_color}[{status_text}] {filepath.name}{END}")
        else:
            print(f"{RED}[FAIL] {filepath.name} - Critical issues found{END}")
        
        return not hard_fail
        
    except Exception as e:
        print(f"{RED}[FAIL] {filepath.name}: Unexpected error: {e}{END}")
        return False

def main(paths):
    """Main QC runner"""
    print(f"{BLUE}DSBL QC Batch Validator{END}")
    print("=" * 50)
    
    all_ok = True
    passed = 0
    failed = 0
    
    for path_str in paths:
        filepath = pathlib.Path(path_str)
        if not filepath.exists():
            print(f"{RED}[ERROR] File not found: {filepath}{END}")
            failed += 1
            all_ok = False
            continue
            
        if qc_file(filepath):
            passed += 1
        else:
            failed += 1
            all_ok = False
    
    print("\n" + "=" * 50)
    print(f"QC Summary: {GREEN}{passed} PASSED{END}, {RED}{failed} FAILED{END}")
    
    if all_ok:
        print(f"{GREEN}✓ All files passed QC validation{END}")
        return 0
    else:
        print(f"{RED}✗ Some files failed validation - check logs above{END}")
        return 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file1.jsonl> [file2.jsonl] ...")
        print("Example: python qc/qc_batch.py exp_output/published_data/batch_*/events/*.jsonl")
        print("         python qc/qc_batch.py exp_output/production_test/events/malice_*.jsonl")
        sys.exit(1)
    
    sys.exit(main(sys.argv[1:]))