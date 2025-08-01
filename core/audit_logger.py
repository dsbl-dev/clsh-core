# SPDX-FileCopyrightText: 2025 DSBL-Dev contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Audit logging for experiments.
Provides JSONL file logging and real-time console output.
"""

import json
from datetime import datetime
from typing import Dict, List

class AuditLogger:
    """Handles structured audit logging"""
    
    def __init__(self, experiment_name: str = "umbrix", run_id: int = None, 
                 total_runs: int = None, tickets: int = None, metrics_data_mode: bool = False,
                 console_output: bool = True):
        self.audit_log: List[Dict] = []
        # Alias for backward compatibility
        self.events = self.audit_log
        
        # Metrics mode settings
        self.metrics_data_mode = metrics_data_mode
        self.metrics_gate_decisions = metrics_data_mode
        self.metrics_vote_calculations = metrics_data_mode
        self.metrics_state_changes = metrics_data_mode
        self.metrics_timing = metrics_data_mode
        
        # Console output control - when False, no console messages to terminal
        self.console_output = console_output
        
        # Timing diagnostics for performance analysis
        self.timing_checkpoints = {}
        self.ticket_start_time = None
        
        # Track timing for duration calculation
        self.start_time = datetime.now()
        
        # Create JSONL audit file with structured naming
        # Format: malice_250607_16h32m45s_r05-30_t40_duration.jsonl
        timestamp = datetime.now().strftime("%y%m%d_%Hh%Mm%Ss")
        
        # Build filename parts
        filename_parts = [experiment_name, timestamp]
        
        if run_id is not None and total_runs is not None:
            filename_parts.append(f"r{run_id:02d}-{total_runs:02d}")
        
        if tickets is not None:
            filename_parts.append(f"t{tickets}")
            
        # Will add duration when finalizing
        base_filename = "_".join(filename_parts)
        self.audit_file_base = base_filename
        # Store logs in exp_output/ directory for better organization
        self.audit_file = f"exp_output/{base_filename}.jsonl"
        
        # Create metrics file (always enabled for research data)
        self.metrics_file = f"exp_output/{base_filename}_metrics.jsonl"
        # Note: metrics directory creation and final path setting is handled by calling code 
        # (main.py or batch_runner_parallel.py) to ensure correct placement in subdirectories
        
    def log_event(self, event_type: str, details: Dict):
        """Log structured audit event to both memory and JSONL file."""
        audit_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "event_type": event_type,
            "details": details
        }
        self.audit_log.append(audit_entry)
        
        # Write to JSONL file for reviewers
        try:
            with open(self.audit_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(audit_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"[AUDIT ERROR]: Failed to write to {self.audit_file}: {e}")
        
        # Console output with filtering support
        self._print_event(event_type, details)
    
    def log_metric_event(self, event_type: str, details: Dict):
        """Log metrics events to separate metrics file (always enabled)."""
        if not self.metrics_file:
            return
            
        metrics_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "event_type": event_type,
            "details": details
        }
        
        # Write to metrics JSONL file
        try:
            with open(self.metrics_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metrics_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"[METRICS ERROR]: Failed to write to {self.metrics_file}: {e}")
        
        # Console output for metrics events (only if metrics_data_mode enabled)
        if self.metrics_data_mode:
            self._print_event(event_type, details)
    
    def _print_event(self, event_type: str, details: Dict):
        """Print audit event to console with filtering."""
        # Skip all console output if console_output is disabled
        if not self.console_output:
            return
            
        # Always show critical events
        always_show = ["MESSAGE_SENT", "VOTE_PROCESSED", "USER_PROMOTED", "USER_DEMOTED", "EXPERIMENT_SUMMARY"]
        
        # Metrics-only events - no console output, only JSONL
        metrics_only = ["GATE_METRIC", "VOTE_METRIC", "STATE_METRIC", "TIMING_METRIC"]
        
        # Never show metrics events in console
        if event_type in metrics_only:
            return
        
        # Show regular events unless explicitly metrics-only
        if event_type not in always_show and event_type not in metrics_only and not self.metrics_data_mode:
            # In non-metrics mode, only show GATE_DECISION events
            if event_type != "GATE_DECISION":
                return
        
        # Color-coded output for different event types
        if event_type == "GATE_DECISION":
            gate = details.get("gate", "")
            result = details.get("result", "")
            
            if gate == "civil":
                if result == "BLOCKED":
                    print(f"[🛡️ CIVIL_BLOCKED]: {details.get('reason', '')}")
                else:
                    print(f"[🛡️ CIVIL_ALLOWED]: {details.get('reason', '')}")
            elif gate == "sec_clean":
                if result == "BLOCKED":
                    print(f"[👮 SEC_BLOCKED]: {details.get('reason', '')}")
                else:
                    print(f"[👮 SEC_ALLOWED]: {details.get('reason', '')}")
        
        # Metrics events are handled above - no console output
        
        else:
            # Standard events
            print(f"[AUDIT {event_type}]: {details}")
    
    def log_gate_metric(self, metric_data: Dict):
        """Log detailed gate decision metrics information."""
        if not self.metrics_gate_decisions:
            return
            
        self.log_metric_event("GATE_METRIC", metric_data)
    
    def log_vote_metric(self, metric_data: Dict):
        """Log detailed vote calculation metrics information."""
        if not self.metrics_vote_calculations:
            return
            
        self.log_metric_event("VOTE_METRIC", metric_data)
    
    def log_state_metric(self, metric_data: Dict):
        """Log detailed state change metrics information."""
        if not self.metrics_state_changes:
            return
            
        self.log_metric_event("STATE_METRIC", metric_data)
    
    def start_ticket_timing(self, ticket_number: int):
        """Start timing a new ticket generation cycle."""
        if not self.metrics_timing:
            return
        
        import time
        self.ticket_start_time = time.time()
        self.timing_checkpoints = {}
        
        # Skip detailed ticket_start logging to reduce spam
        pass
    
    def log_timing_checkpoint(self, checkpoint_name: str, details: str = None):
        """Log a timing checkpoint during ticket processing."""
        if not self.metrics_timing or self.ticket_start_time is None:
            return
        
        import time
        current_time = time.time()
        elapsed = current_time - self.ticket_start_time
        
        # Calculate time since last checkpoint
        last_checkpoint_time = self.timing_checkpoints.get('last_time', self.ticket_start_time)
        delta = current_time - last_checkpoint_time
        
        self.timing_checkpoints[checkpoint_name] = current_time
        self.timing_checkpoints['last_time'] = current_time
        
        # Only log significant timing events to reduce spam - exclude sleep and empty messages
        is_sleep = checkpoint_name.startswith('sleep')
        is_empty_message = 'message length: 0' in (details or '')
        
        if not is_sleep and not is_empty_message and (delta > 1.0 or checkpoint_name in ['message_gen_end', 'gate_processing_end']):
            timing_details = {
                "type": "checkpoint",
                "checkpoint": checkpoint_name,
                "elapsed_total_seconds": int(elapsed),
                "delta_seconds": int(delta),
                "details": details
            }
            self.log_metric_event("TIMING_METRIC", timing_details)
        
        # Only show critical slow operations in console (>2s) - exclude expected sleep delays
        if delta > 2.0 and not checkpoint_name.startswith('sleep'):
            print(f"⏱️ SLOW: {checkpoint_name} took {delta:.1f}s (total: {elapsed:.1f}s) - {details or ''}")
    
    def finish_ticket_timing(self, ticket_number: int, total_messages: int = None):
        """Finish timing for current ticket and summarize."""
        if not self.metrics_timing or self.ticket_start_time is None:
            return
        
        import time
        total_time = time.time() - self.ticket_start_time
        
        # Only log ticket completion if it took significant time
        if total_time > 3.0:
            summary = {
                "type": "ticket_complete",
                "ticket": ticket_number,
                "total_seconds": int(total_time),
                "messages_processed": total_messages
            }
            self.log_metric_event("TIMING_METRIC", summary)
        
        # Reset for next ticket
        self.ticket_start_time = None
        self.timing_checkpoints = {}
        
        # Console summary - only show slow tickets
        if total_time > 10.0:
            print(f"⚠️ SLOW TICKET #{ticket_number}: {total_time:.1f}s total")
    
    def get_events_by_type(self, event_type: str) -> List[Dict]:
        """Filter audit log by event type for analysis."""
        return [event for event in self.audit_log if event["event_type"] == event_type]
    
    def get_stats(self) -> Dict:
        """Get summary statistics of logged events."""
        event_counts = {}
        for event in self.audit_log:
            event_type = event["event_type"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        return {
            "total_events": len(self.audit_log),
            "event_types": event_counts,
            "audit_file": self.audit_file
        }
    
    def log_symbol_interpretation(self, symbol_type: str, symbol_content: str, 
                                 interpreter: str, context: Dict, interpretation: Dict):
        """
        Log symbol interpretation for Symbol Journey Timeline analysis.
        
        Args:
            symbol_type: Type of symbol (e.g., 'GATE', 'VOTE', 'CIVIL')
            symbol_content: The actual symbol content (e.g., 'sec_clean', 'promote_alice')
            interpreter: Who/what interpreted the symbol (e.g., 'gate_processor', 'vote_system', 'mallory')
            context: Current context affecting interpretation (ticket, user status, reputation, etc.)
            interpretation: The actual interpretation result (decision, action taken, etc.)
        """
        symbol_event = {
            "symbol_type": symbol_type,
            "symbol_content": symbol_content,
            "interpreter": interpreter,
            "context": context,
            "interpretation": interpretation,
            "ticket": context.get("ticket"),
            "timestamp_detailed": datetime.now().isoformat()
        }
        
        self.log_event("SYMBOL_INTERPRETATION", symbol_event)
    
    def log_agent_message(self, sender: str, message: str, ticket_id: int):
        """
        Log agent conversations to metrics log for post-batch conversation viewing.
        
        Args:
            sender: Name of the agent sending the message
            message: The actual message content
            ticket_id: Current ticket number for context
        """
        if not self.metrics_data_mode or not self.metrics_file:
            return
        
        conversation_event = {
            "sender": sender,
            "message": message,
            "ticket_id": ticket_id,
            "run_id": getattr(self, 'run_id', None),
            "timestamp_detailed": datetime.now().isoformat()
        }
        
        self.log_metric_event("AGENT_MESSAGE", conversation_event)
    
    def finalize_with_duration(self):
        """Rename audit and metrics files to include final duration."""
        try:
            import os
            duration = datetime.now() - self.start_time
            duration_minutes = int(duration.total_seconds() / 60)
            duration_seconds = int(duration.total_seconds() % 60)
            
            # Format duration: 34m15s -> d34m or 2m05s -> d2m  
            if duration_minutes > 0:
                duration_str = f"d{duration_minutes}m"
            else:
                duration_str = f"d{duration_seconds}s"
            
            # Rename main audit file - preserve existing directory structure
            if os.path.exists(self.audit_file):
                # Extract directory from current audit_file path
                current_dir = os.path.dirname(self.audit_file)
                final_filename = f"{current_dir}/{self.audit_file_base}_{duration_str}.jsonl"
                os.rename(self.audit_file, final_filename)
                self.audit_file = final_filename
                print(f"📝 Audit log finalized: {final_filename}")
            
            # Rename metrics file if it exists - preserve existing directory structure
            if self.metrics_file and os.path.exists(self.metrics_file):
                # Extract directory from current metrics_file path
                current_metrics_dir = os.path.dirname(self.metrics_file)
                metrics_final_filename = f"{current_metrics_dir}/{self.audit_file_base}_metrics_{duration_str}.jsonl"
                os.rename(self.metrics_file, metrics_final_filename)
                self.metrics_file = metrics_final_filename
                print(f"📊 Metrics log finalized: {metrics_final_filename}")
            
        except Exception as e:
            print(f"[AUDIT WARNING]: Could not rename audit file: {e}")
            
        return self.audit_file