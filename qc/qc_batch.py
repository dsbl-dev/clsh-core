#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 DSBL-Dev contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
QC script for batches.
Checks data integrity and experiment validity for JSONL audit logs.
"""

import sys
import json
import pathlib
from collections import defaultdict
import re

# Color codes for output
RED = "\033[91m"
YELLOW = "\033[93m" 
GREEN = "\033[92m"
BLUE = "\033[94m"
END = "\033[0m"

def extract_run_metadata(filepath: pathlib.Path) -> dict:
    """Extract metadata from filename: malice_YYMMDD_HHhMMmSSs_pPID_rXX_tZZ_dMM"""
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
        # Read JSONL file line by line for better error handling
        events = []
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    events.append(event)
                except json.JSONDecodeError as e:
                    print(f"  {RED}[FAIL] Line {line_num}: Invalid JSON - {e}{END}")
                    hard_fail = True
                    continue
        
        if not events:
            print(f"  {RED}[FAIL] No valid events found{END}")
            return False
        
        # Extract run metadata
        metadata = extract_run_metadata(filepath)
        expected_tickets = metadata.get('expected_tickets')
        
        # === TICKET COVERAGE CHECK ===
        
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
        
        if tickets:
            min_ticket = min(tickets)
            max_ticket = max(tickets)
            ticket_range = max_ticket - min_ticket + 1
            coverage = len(tickets) / ticket_range * 100
            
            print(f"  Tickets: {len(tickets)} unique, range #{min_ticket}-#{max_ticket} (coverage: {coverage:.1f}%)")
            
            if coverage < 60:
                print(f"  {RED}[FAIL] Low ticket coverage: {coverage:.1f}% < 60%{END}")
                hard_fail = True
            elif coverage < 70:
                print(f"  {YELLOW}[WARN] Moderate ticket coverage: {coverage:.1f}%{END}")
                warn = True
        else:
            print(f"  {RED}[FAIL] No valid tickets found{END}")
            hard_fail = True
        
        # Check against expected tickets if available
        if expected_tickets and latest_ticket:
            progress_pct = (latest_ticket / expected_tickets) * 100
            print(f"  Progress: {latest_ticket}/{expected_tickets} tickets ({progress_pct:.1f}%)")
            
            if progress_pct < 80:
                print(f"  {YELLOW}[WARN] Experiment may be incomplete: {progress_pct:.1f}% < 80%{END}")
                warn = True
        
        # === MESSAGE ID MONOTONICITY CHECK ===
        
        message_ids = []
        for event in events:
            msg_id = event.get('message_id')
            if msg_id is not None:
                message_ids.append(msg_id)
        
        if message_ids:
            # Check for duplicates
            if len(message_ids) != len(set(message_ids)):
                print(f"  {RED}[FAIL] Duplicate message IDs detected{END}")
                hard_fail = True
            
            # Check monotonicity
            non_monotonic = 0
            for i in range(1, len(message_ids)):
                if message_ids[i] <= message_ids[i-1]:
                    non_monotonic += 1
            
            if non_monotonic > 0:
                non_monotonic_pct = (non_monotonic / len(message_ids)) * 100
                print(f"  Message IDs: {non_monotonic} non-monotonic ({non_monotonic_pct:.1f}%)")
                
                if non_monotonic_pct > 10:
                    print(f"  {RED}[FAIL] High non-monotonic message IDs: {non_monotonic_pct:.1f}% > 10%{END}")
                    hard_fail = True
                elif non_monotonic_pct > 5:
                    print(f"  {YELLOW}[WARN] Some non-monotonic message IDs: {non_monotonic_pct:.1f}%{END}")
                    warn = True
        
        # === STATUS CONSISTENCY CHECK ===
        
        # Track user statuses
        user_statuses = defaultdict(lambda: 'REGULAR')
        status_violations = 0
        
        for event in events:
            event_type = event.get('event_type')
            details = event.get('details', {})
            
            # Update user statuses
            if event_type == 'STATUS_CHANGE':
                username = details.get('username')
                change_type = details.get('type')
                if username and change_type in ['PROMOTION', 'DEMOTION']:
                    if change_type == 'PROMOTION':
                        user_statuses[username] = 'BINDER'
                    else:  # DEMOTION
                        user_statuses[username] = 'DEMOTED'
            
            # Check voting violations
            elif event_type == 'VOTE_PROCESSING':
                voter = details.get('voter')
                if voter and user_statuses[voter] == 'DEMOTED':
                    status_violations += 1
        
        if status_violations > 0:
            print(f"  {RED}[FAIL] Status violations: {status_violations} votes from demoted users{END}")
            hard_fail = True
        
        # === BINDER EMERGENCE CHECK ===
        
        promotions = [e for e in events 
                     if e.get('event_type') == 'STATUS_CHANGE' 
                     and e.get('details', {}).get('type') == 'PROMOTION']
        
        if not promotions:
            if len(tickets) > 30:  # Only warn for longer experiments
                print(f"  {YELLOW}[WARN] No BINDER promotions detected (experiment may be too short){END}")
                warn = True
        else:
            promoted_users = [p.get('details', {}).get('username') for p in promotions]
            print(f"  BINDER promotions: {len(promotions)} ({', '.join(promoted_users)})")
        
        # === GATE CONSISTENCY CHECK ===
        
        gate_inconsistencies = 0
        for event in events:
            if (event.get('event_type') == 'GATE_DECISION' and
                event.get('details', {}).get('result') == 'ALLOWED'):
                # Check if there's a subsequent penalty for the same user/content
                user = event.get('details', {}).get('user')
                if user:
                    # Look for reputation penalties shortly after
                    # This is a simplified check
                    pass
        
        # === ADAPTIVE RESPONSE CHECK ===
        
        immune_adjustments = [e for e in events 
                             if e.get('event_type') in ['IMMUNE_RESPONSE_ADJUSTMENT', 'IMMUNE_FREQUENCY_ADJUSTMENT']]
        
        if immune_adjustments:
            print(f"  Adaptive responses: {len(immune_adjustments)} immune system adjustments detected")
        
        # === FINAL RESULT ===
        
        if hard_fail:
            print(f"  {RED}[FAIL] Critical issues detected{END}")
            return False
        elif warn:
            print(f"  {YELLOW}[PASS] Passed with warnings{END}")
            return True
        else:
            print(f"  {GREEN}[PASS] All checks passed{END}")
            return True
            
    except Exception as e:
        print(f"  {RED}[ERROR] QC analysis failed: {e}{END}")
        return False

def main():
    """Main QC runner"""
    
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <logfile1.jsonl> [logfile2.jsonl] ...")
        print(f"Example: {sys.argv[0]} exp_output/batch_name/events/*.jsonl")
        return 1
    
    log_files = []
    for arg in sys.argv[1:]:
        path = pathlib.Path(arg)
        if path.is_file() and path.suffix == '.jsonl':
            log_files.append(path)
        elif path.exists():
            print(f"{YELLOW}[WARN] Skipping non-JSONL file: {arg}{END}")
    
    if not log_files:
        print(f"{RED}[ERROR] No valid JSONL files found{END}")
        return 1
    
    print(f"{BLUE}Batch QC Analysis{END}")
    print(f"Checking {len(log_files)} files...")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for log_file in log_files:
        if qc_file(log_file):
            passed += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    total = passed + failed
    if failed == 0:
        print(f"{GREEN}[SUMMARY] All {total} files passed QC{END}")
        return 0
    elif passed == 0:
        print(f"{RED}[SUMMARY] All {total} files failed QC{END}")
        return 1
    else:
        print(f"{YELLOW}[SUMMARY] {passed}/{total} files passed, {failed} failed{END}")
        return 1

if __name__ == "__main__":
    sys.exit(main())