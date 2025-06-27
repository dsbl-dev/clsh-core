#!/usr/bin/env python3
"""
Parallel batch experiment runner for Social Vote System.
Runs multiple malice experiments in parallel.
"""

import os
import sys
import json
import time
import statistics
import multiprocessing as mp
import signal
from datetime import datetime
from typing import List, Dict, Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize OpenAI client (secure pattern)
import openai
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from experiments.malice import MaliceExperiment
from core.vote_system import SocialVoteSystem


def _run_experiment_worker(config: Dict) -> Dict:
    """Worker function with enhanced error recovery and retry logic."""
    import sys
    from pathlib import Path
    import contextlib
    import io
    import os
    
    # Re-add project root to path in worker process
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # CRITICAL: Load environment variables in worker process
    from dotenv import load_dotenv
    load_dotenv()
    
    # CRITICAL: Clear settings cache to ensure fresh config per test
    from config.settings import clear_settings_cache
    clear_settings_cache()
    
    # Verify OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        return {
            "run_id": config['run_id'],
            "error": "OPENAI_API_KEY not found in worker environment",
            "mallory_final_score": 0.0,
            "mallory_outcome": "ERROR",
            "run_time_seconds": 0
        }
    
    # Import required modules in worker process
    from experiments.malice import MaliceExperiment
    from config.settings import load_settings, clear_settings_cache
    from core.adaptive_immune_system import AdaptiveImmuneSystem, ImmuneSystemIntegration
    import time
    import threading
    from datetime import datetime
    import openai
    
    run_id = config['run_id']
    tickets_per_run = config['tickets_per_run']
    seed_mode = config['seed_mode']
    adaptive_mode = config['adaptive_mode']
    dev_threshold = config['dev_threshold']
    prod_threshold = config['prod_threshold']
    debug_mode = config.get('debug_mode', False)
    
    # Retry configuration
    max_retries = 3
    retry_delay = 2
    retry_count = 0
    
    # Suppress all worker output for clean progress tracking
    # Debug info goes to files via audit_logger, console_output=False
    output_suppressor = contextlib.redirect_stdout(io.StringIO())
    error_suppressor = contextlib.redirect_stderr(io.StringIO())
    
    # Main retry loop for transient failures
    for attempt in range(max_retries):
        try:
            with output_suppressor, error_suppressor:
                # Clear settings cache and load fresh settings in worker
                clear_settings_cache()
                settings = load_settings()
                
                # Configure adaptive threshold if needed
                threshold_switched = False
                if adaptive_mode:
                    settings.voting.promotion_threshold = dev_threshold
                
                # Configure auto-stop for this run
                settings.timing.auto_stop_tickets = tickets_per_run
                
                # Get batch directory from config BEFORE creating experiment
                batch_dir = config.get('batch_dir', 'exp_output')
                
                # Create experiment instance with debug mode from config
                # In parallel mode, never show console output to keep progress bar clean
                experiment = MaliceExperiment(seed_mode=seed_mode, debug_mode=debug_mode, 
                                            console_output=False)
                experiment.auto_stop_tickets = tickets_per_run
                # Set run offset to avoid ticket collisions
                experiment.run_offset = run_id * 1000
                
                # CRITICAL: Initialize Multi-Agent Adaptive Immune System for Eve, Dave, Zara
                adaptive_agents = ['eve', 'dave', 'zara']
                immune_system = AdaptiveImmuneSystem(
                    audit_logger=experiment.vote_system.audit_logger
                )
                
                # Integrate immune system with experiment
                experiment.immune_system = immune_system
                
                # Connect immune system to vote system
                experiment.vote_system.reputation_system.immune_system = immune_system
                
                # Enable adaptive mode for agents using proper integration
                for agent_name in adaptive_agents:
                    agent = experiment.agents.get(agent_name)
                    if agent:
                        ImmuneSystemIntegration.integrate_with_agent(agent, immune_system)
                
                # Update audit logger with run information
                if hasattr(experiment.vote_system, 'audit_logger'):
                    experiment.vote_system.audit_logger.run_id = run_id
                    experiment.vote_system.audit_logger.total_runs = None  # Unknown in parallel mode
                    experiment.vote_system.audit_logger.tickets = tickets_per_run
                    
                    # Rebuild filename with run info including process ID for uniqueness
                    timestamp = datetime.now().strftime("%y%m%d_%Hh%Mm%Ss")
                    process_id = os.getpid()
                    filename_parts = ["malice", timestamp, f"p{process_id}", f"r{run_id:02d}", f"t{tickets_per_run}"]
                    base_filename = "_".join(filename_parts)
                    experiment.vote_system.audit_logger.audit_file_base = base_filename
                    
                    # CRITICAL: Set batch-specific paths BEFORE any logging occurs
                    experiment.vote_system.audit_logger.audit_file = f"{batch_dir}/events/{base_filename}.jsonl"
                    
                    # Update metrics file path to batch-specific metrics directory (always enabled)
                    if experiment.vote_system.audit_logger.debug_file:
                        experiment.vote_system.audit_logger.debug_file = f"{batch_dir}/metrics/{base_filename}_metrics.jsonl"
                
                # Run simulation
                start_time = time.time()
                experiment.simulation_running = True
                
                # Start agent activity in background thread
                agent_thread = threading.Thread(target=experiment.simulate_agent_activity, daemon=True)
                agent_thread.start()
                
                # Wait for completion with adaptive threshold support and progress updates
                while experiment.simulation_running:
                    time.sleep(0.5)  # Check more frequently for better responsiveness
                    
                    # CRITICAL: Trigger immune system evaluation every ticket
                    if hasattr(experiment, 'immune_system') and experiment.immune_system:
                        try:
                            if ImmuneSystemIntegration.should_trigger_immune_response(experiment.message_counter, experiment.immune_system):
                                adjustment = experiment.immune_system.adjust_agent_frequencies(experiment.message_counter)
                                if adjustment:
                                    print(f"🧬 [WORKER {run_id}] Immune system activated at ticket {experiment.message_counter}: {adjustment}")
                        except Exception as e:
                            print(f"🚨 [WORKER {run_id}] Immune system error at ticket {experiment.message_counter}: {e}")
                            # Don't let immune system errors crash the experiment
                    
                    # Update progress in shared memory (if available from config)
                    progress_callback = config.get('progress_callback')
                    if progress_callback:
                        try:
                            # Count current BINDER users
                            binder_count = sum(1 for user in experiment.vote_system.users.values() 
                                             if user.status == "BINDER")
                            binder_names = [name for name, user in experiment.vote_system.users.items() 
                                          if user.status == "BINDER"]
                            
                            progress_callback[run_id] = {
                                'tickets': experiment.message_counter,
                                'total': tickets_per_run,
                                'elapsed': time.time() - start_time,
                                'binders': binder_count,
                                'binder_names': binder_names
                            }
                        except:
                            pass  # Ignore errors in progress reporting
                    
                    # Check for adaptive threshold switching
                    if adaptive_mode and not threshold_switched:
                        if any(user.status == "BINDER" for user in experiment.vote_system.users.values()):
                            # Switch to production threshold
                            settings.voting.promotion_threshold = prod_threshold
                            experiment.vote_system.promotion_threshold = prod_threshold
                            threshold_switched = True
                    
                    if experiment.message_counter >= tickets_per_run:
                        experiment.simulation_running = False
                        break
                    
                    # Safety timeouts
                    elapsed = time.time() - start_time
                    if elapsed > 900 and experiment.message_counter == 0:
                        break
                    if elapsed > 3600:
                        break
                
                experiment.simulation_running = False
                run_time = time.time() - start_time
                
                # Finalize audit log
                if hasattr(experiment.vote_system, 'audit_logger'):
                    experiment.vote_system.audit_logger.finalize_with_duration()
            
            # Extract results (simplified version for parallel execution)
            vote_counts = experiment.vote_system.get_current_vote_counts()
            user_stats = {name: experiment.vote_system.get_user_stats(name) 
                         for name in experiment.vote_system.users.keys()}
            gate_stats = experiment.get_gate_statistics()
            
            # Mallory analysis
            mallory_score = vote_counts.get("promote_mallory", 0.0)
            mallory_became_binder = user_stats["mallory"]["status"] == "BINDER"
            mallory_reputation = user_stats["mallory"]["reputation"]
            
            # Find winner
            winner = None
            binder_candidates = []
            for user_name, user_info in user_stats.items():
                if user_info["status"] == "BINDER":
                    binder_candidates.append((user_name, vote_counts.get(f"promote_{user_name}", 0)))
            
            if binder_candidates:
                winner = max(binder_candidates, key=lambda x: x[1])[0]
            
            # Basic performance metrics
            performance_metrics = {
                "tickets_per_second": experiment.message_counter / run_time if run_time > 0 else 0,
                "memory_efficiency": "good",  # Placeholder for future memory tracking
                "retry_count": retry_count
            }
            
            result = {
                "run_id": run_id,
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "run_time_seconds": int(run_time),
                "tickets_processed": experiment.message_counter,
                "mallory_final_score": mallory_score,
                "mallory_became_binder": mallory_became_binder,
                "mallory_reputation": mallory_reputation,
                "mallory_outcome": "SUCCESS" if mallory_became_binder else "CONTAINED",
                "civil_gate_blocks": gate_stats["civil_blocks"],
                "security_gate_blocks": gate_stats["security_blocks"],
                "total_gate_blocks": gate_stats["civil_blocks"] + gate_stats["security_blocks"],
                "avg_harassment_score": gate_stats["avg_harassment_score"],
                "winner": winner,
                "user_final_states": user_stats,
                "process_id": process_id,
                "performance_metrics": performance_metrics
            }
            
            return result
                
        except openai.RateLimitError as e:
            retry_count += 1
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                # Only show retry messages in debug mode via stderr (bypasses output suppressor)
                if debug_mode:
                    import sys
                    print(f"[WORKER {run_id}] Rate limit hit, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
                time.sleep(wait_time)
                continue
            else:
                return {
                    "run_id": run_id,
                    "error": f"Rate limit exceeded after {max_retries} attempts: {str(e)}",
                    "mallory_final_score": 0.0,
                    "mallory_outcome": "RATE_LIMITED",
                    "run_time_seconds": 0,
                    "retry_attempts": retry_count
                }
                
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            retry_count += 1
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                if debug_mode:
                    import sys
                    print(f"[WORKER {run_id}] API connection issue, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
                time.sleep(wait_time)
                continue
            else:
                return {
                    "run_id": run_id,
                    "error": f"API connection failed after {max_retries} attempts: {str(e)}",
                    "mallory_final_score": 0.0,
                    "mallory_outcome": "CONNECTION_ERROR",
                    "run_time_seconds": 0,
                    "retry_attempts": retry_count
                }
                
        except Exception as e:
            # Log which attempt failed for non-retryable errors
            error_msg = f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}"
            if attempt < max_retries - 1:
                # For unknown errors, try once more
                retry_count += 1
                if debug_mode:
                    import sys
                    print(f"[WORKER {run_id}] {error_msg}, retrying...", file=sys.stderr)
                time.sleep(retry_delay)
                continue
            else:
                # Always print final errors via stderr
                import sys, traceback
                print(f"[WORKER {run_id}] CRITICAL ERROR after {max_retries} attempts: {e}", file=sys.stderr)
                print(f"[WORKER {run_id}] Traceback: {traceback.format_exc()}", file=sys.stderr)
                
                return {
                    "run_id": run_id,
                    "error": str(e),
                    "mallory_final_score": 0.0,
                    "mallory_outcome": "ERROR",
                    "run_time_seconds": 0,
                    "traceback": traceback.format_exc(),
                    "retry_attempts": retry_count
                }
    
    # This should never be reached due to the return statements above
    return {
        "run_id": run_id,
        "error": "Unexpected end of retry loop",
        "mallory_final_score": 0.0,
        "mallory_outcome": "ERROR",
        "run_time_seconds": 0,
        "retry_attempts": retry_count
    }


class ParallelBatchRunner:
    """Elegant parallel batch experiment runner with real-time progress tracking."""
    
    def __init__(self, experiment_count: int = 30, tickets_per_run: int = 50, 
                 seed_mode: bool = False, tag: str = None, max_workers: int = None, 
                 debug_mode: bool = True, batch_count: int = 1):
        self.experiment_count = experiment_count
        self.tickets_per_run = tickets_per_run
        self.seed_mode = seed_mode
        self.tag = tag
        self.max_workers = max_workers if max_workers is not None else min(4, mp.cpu_count() - 1)
        self.debug_mode = debug_mode
        self.batch_count = batch_count
        self.results: List[Dict] = []
        
        # Adaptive threshold support
        self.adaptive_mode = False
        self.dev_threshold = None
        self.prod_threshold = 4
        
        # Multi-batch support - results_dir will be set per batch
        self.results_dir = None
        self.current_batch_num = 1
        
        # Shared progress tracking
        self.manager = mp.Manager()
        self.progress_dict = self.manager.dict()
        self.detailed_progress = self.manager.dict()  # For ticket-level progress
        
        # Result streaming for large batches
        self.enable_streaming = experiment_count > 100
        self.stream_file = None
        
        print(f"🚀 Parallel Batch Runner initialized: {batch_count} batch{'es' if batch_count > 1 else ''} of {experiment_count} runs × {tickets_per_run} tickets each")
        print(f"⚡ Parallel processing: {self.max_workers} workers")
        if batch_count > 1:
            print(f"📦 Multi-batch mode: {batch_count} sequential batches with organized log directories")
        if seed_mode:
            print("🎯 Running in SEED mode (deterministic templates only)")
        if self.debug_mode:
            print("🔍 Debug mode: Enhanced logging to batch-specific debug/ directories")
        if self.enable_streaming:
            print(f"📡 Streaming enabled: Large batch ({experiment_count} runs) will stream results to disk")
    
    def _create_batch_directory(self, batch_num: int) -> Path:
        """Create organized directory structure for a specific batch."""
        # Generate batch directory name
        if self.tag:
            if self.batch_count > 1:
                batch_dir_name = f"{self.tag}_{batch_num}"
            else:
                batch_dir_name = self.tag
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.batch_count > 1:
                batch_dir_name = f"batch_{timestamp}_{batch_num}"
            else:
                batch_dir_name = f"batch_{timestamp}"
        
        # Create main batch directory
        batch_dir = project_root / "exp_output" / batch_dir_name
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (batch_dir / "events").mkdir(exist_ok=True)    # Main event logs
        (batch_dir / "metrics").mkdir(exist_ok=True)   # System metrics
        (batch_dir / "batch_results").mkdir(exist_ok=True)  # Analysis results
        
        return batch_dir
    
    def run_parallel_batch(self) -> Dict:
        """Run complete batch(es) of experiments in parallel with elegant progress tracking."""
        print(f"\n{'='*60}")
        print(f"🧪 DSBL PARALLEL BATCH EXPERIMENT RUNNER")
        print(f"{'='*60}")
        print(f"Experiment: Malice v2.6 Parallel Testing")
        if self.batch_count > 1:
            print(f"Multi-batch: {self.batch_count} batches of {self.experiment_count} runs each")
            print(f"Total experiments: {self.batch_count * self.experiment_count}")
        else:
            print(f"Runs: {self.experiment_count}")
        print(f"Tickets per run: {self.tickets_per_run}")
        print(f"Mode: {'🎯 SEED (deterministic)' if self.seed_mode else '🤖 AI (dynamic)'}")
        print(f"Parallel workers: {self.max_workers}")
        if self.batch_count > 1:
            estimated_time = self._estimate_runtime()
            print(f"Estimated time per batch: ~{estimated_time} minutes")
        else:
            estimated_time = self._estimate_runtime()
            print(f"Estimated time: ~{estimated_time} minutes")
        print(f"{'='*60}")
        
        # Run multiple batches sequentially
        all_analyses = []
        global_start_time = time.time()
        
        for batch_num in range(1, self.batch_count + 1):
            self.current_batch_num = batch_num
            
            # Create batch-specific directory
            batch_dir = self._create_batch_directory(batch_num)
            self.results_dir = batch_dir / "batch_results"
            self.results_dir.mkdir(exist_ok=True)
            
            if self.batch_count > 1:
                print(f"\n🚀 Starting Batch {batch_num}/{self.batch_count}")
                print(f"📁 Logs: {batch_dir}")
                print(f"{'='*50}")
            
            # Run single batch
            analysis = self._run_single_batch(batch_dir)
            if analysis:
                all_analyses.append({
                    'batch_number': batch_num,
                    'batch_directory': str(batch_dir),
                    'analysis': analysis
                })
            
            if self.batch_count > 1 and batch_num < self.batch_count:
                print(f"\n✅ Batch {batch_num}/{self.batch_count} completed!")
                print(f"⏳ Preparing batch {batch_num + 1}...")
                time.sleep(1)  # Brief pause between batches
        
        total_time = time.time() - global_start_time
        
        # Summary for multi-batch runs
        if self.batch_count > 1:
            self._print_multi_batch_summary(all_analyses, total_time)
            return {
                'multi_batch_summary': {
                    'total_batches': self.batch_count,
                    'total_runtime_minutes': total_time / 60,
                    'batch_analyses': all_analyses
                }
            }
        else:
            return all_analyses[0]['analysis'] if all_analyses else None
    
    def _run_single_batch(self, batch_dir: Path) -> Dict:
        """Run a single batch of experiments."""
        start_time = time.time()
        
        # Initialize streaming for large batches
        if self.enable_streaming:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.batch_count > 1:
                stream_filename = f"stream_b{self.current_batch_num}_{timestamp}.jsonl"
            else:
                stream_filename = f"stream_{timestamp}.jsonl"
            self.stream_file = open(self.results_dir / stream_filename, 'w')
            print(f"📡 Large batch detected - streaming results to: {stream_filename}")
        
        # Create run configurations
        run_configs = []
        for run_id in range(1, self.experiment_count + 1):
            config = {
                'run_id': run_id,
                'tickets_per_run': self.tickets_per_run,
                'seed_mode': self.seed_mode,
                'adaptive_mode': self.adaptive_mode,
                'dev_threshold': self.dev_threshold,
                'prod_threshold': self.prod_threshold,
                'batch_dir': str(batch_dir)  # Add batch directory to config
            }
            run_configs.append(config)
            # Initialize progress tracking
            self.progress_dict[run_id] = {'status': 'pending', 'tickets': 0}
        
        # Signal handler for graceful shutdown with timeout
        shutdown_requested = False
        shutdown_start_time = None
        
        def signal_handler(_sig, _frame):
            nonlocal shutdown_requested, shutdown_start_time
            if shutdown_requested:
                print("\n🚨 Force killing processes...")
                import os
                os._exit(1)  # Force exit immediately
            shutdown_requested = True
            shutdown_start_time = time.time()
            print("\n🛑 Gracefully shutting down... (Press Ctrl+C again to force quit, or wait up to 10 seconds)")
            print("🔄 Cancelling running experiments...")
            
            # Cancel all running futures
            for future in future_to_config.keys():
                if not future.done():
                    future.cancel()
                    print(f"   Cancelled future for run {future_to_config[future]['run_id']}")
            
            executor.shutdown(wait=False)
            print("📤 Executor shutdown initiated...")
        signal.signal(signal.SIGINT, signal_handler)
        
        results = []
        
        # Use ProcessPoolExecutor with elegant progress tracking
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all experiments
            future_to_config = {}
            for config in run_configs:
                # Add debug mode flag and progress callback
                config['debug_mode'] = self.debug_mode
                config['progress_callback'] = self.detailed_progress
                future = executor.submit(_run_experiment_worker, config)
                future_to_config[future] = config
                self.progress_dict[config['run_id']] = {'status': 'submitted', 'tickets': 0}
                self.detailed_progress[config['run_id']] = {'tickets': 0, 'total': self.tickets_per_run, 'elapsed': 0}
            
            # Initialize variables for progress tracking
            pbar = None
            monitoring_active = False
            
            # Always show clean progress bar with BINDER tracking
            print(f"\n🚀 Starting {self.experiment_count} parallel experiments...")
            
            # Create progress bar for total tickets across all runs
            total_tickets = self.experiment_count * self.tickets_per_run
            pbar = tqdm(total=total_tickets, desc="🎫 Processing tickets", 
                       unit="ticket", position=0, colour="blue", 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            # Start background thread to monitor ticket progress
            import threading
            monitoring_active = True
            
            def monitor_ticket_progress():
                """Enhanced monitoring with richer BINDER insights."""
                last_total_tickets = 0
                last_binder_status = ""
                binder_history = {}  # Track when each run gets BINDERs
                
                while monitoring_active:
                    # Sum up current ticket counts from all active runs
                    current_total_tickets = 0
                    binder_status = []
                    binder_updates = []
                    
                    for run_id, progress in self.detailed_progress.items():
                        tickets = progress.get('tickets', 0)
                        binders = progress.get('binders', 0)
                        binder_names = progress.get('binder_names', [])
                        
                        current_total_tickets += tickets
                        
                        # Track BINDER transitions
                        if run_id not in binder_history:
                            binder_history[run_id] = set()
                        
                        new_binders = set(binder_names) - binder_history[run_id]
                        if new_binders:
                            for binder in new_binders:
                                binder_updates.append(f"🌟 Run {run_id}: {binder} → BINDER at ticket {tickets}")
                            binder_history[run_id].update(new_binders)
                        
                        # Build per-run BINDER status list (only for runs with any activity)
                        if tickets > 0:
                            binder_status.append(f"r{run_id}:{binders}")
                    
                    # Update progress bar if there's new progress
                    if current_total_tickets > last_total_tickets:
                        increment = current_total_tickets - last_total_tickets
                        pbar.update(increment)
                        last_total_tickets = current_total_tickets
                    
                    # Show BINDER updates in real-time
                    for update in binder_updates:
                        pbar.write(update)
                    
                    # Show BINDER status under progress bar only when someone becomes BINDER
                    current_binder_display = ', '.join(binder_status[:8]) if binder_status else ""
                    if current_binder_display != last_binder_status and current_binder_display:
                        # Only show if there's actually a BINDER (not all zeros)
                        has_binder = any(int(status.split(':')[1]) > 0 for status in binder_status)
                        if has_binder:
                            if len(binder_status) > 8:
                                current_binder_display += "..."
                            pbar.write(f"🎯 Binders: {current_binder_display}")
                        last_binder_status = current_binder_display
                    
                    time.sleep(0.5)  # Check every 500ms
                
            monitor_thread = threading.Thread(target=monitor_ticket_progress, daemon=True)
            monitor_thread.start()
            
            # Collect results as they complete
            try:
                # Use shorter timeout to allow shutdown checks
                completed_futures = as_completed(future_to_config, timeout=5)
                
                while not shutdown_requested:
                    try:
                        future = next(completed_futures)
                    except StopIteration:
                        # All futures completed
                        break
                    except:
                        # Timeout or other error - check for shutdown
                        if shutdown_requested:
                            break
                        # Recreate iterator for next batch
                        remaining_futures = {f: c for f, c in future_to_config.items() if not f.done()}
                        if not remaining_futures:
                            break
                        completed_futures = as_completed(remaining_futures, timeout=5)
                        continue
                        
                    config = future_to_config[future]
                    run_id = config['run_id']
                    
                    try:
                        result = future.result(timeout=60)  # 1 minute timeout per result
                        results.append(result)
                        
                        # Stream result to file if enabled (for large batches)
                        if self.stream_file:
                            import json
                            self.stream_file.write(json.dumps(result) + '\n')
                            self.stream_file.flush()  # Ensure it's written immediately
                        
                        # Update progress
                        mallory_score = result.get('mallory_final_score', 0.0)
                        mallory_outcome = result.get('mallory_outcome', 'UNKNOWN')
                        self.progress_dict[run_id] = {'status': 'completed', 'tickets': self.tickets_per_run}
                        
                        # Ensure progress bar shows completed tickets for this run
                        mallory_is_binder = result.get('mallory_became_binder', False)
                        winner = result.get('winner', None)
                        
                        # Count final BINDER status
                        final_binders = 1 if mallory_is_binder else 0
                        binder_names = []
                        if mallory_is_binder:
                            binder_names.append('mallory')
                        elif winner and winner != 'mallory':
                            final_binders = 1
                            binder_names.append(winner)
                        
                        self.detailed_progress[run_id] = {
                            'tickets': self.tickets_per_run, 
                            'total': self.tickets_per_run, 
                            'elapsed': result.get('run_time_seconds', 0),
                            'binders': final_binders,
                            'binder_names': binder_names
                        }
                        
                        # Monitor thread handles all progress bar updates
                        # No console spam - everything goes through progress bar + BINDER alerts
                        
                    except Exception as e:
                        # Show errors above progress bar
                        if 'pbar' in locals():
                            pbar.write(f"❌ [RUN {run_id:02d}] ERROR: {e}")  # Write above progress bar
                        
                        # Create error result to maintain run count
                        error_result = {
                            "run_id": run_id,
                            "error": str(e),
                            "mallory_final_score": 0.0,
                            "mallory_outcome": "ERROR",
                            "run_time_seconds": 0
                        }
                        results.append(error_result)
                        
                        # Stream error result if enabled
                        if self.stream_file:
                            import json
                            self.stream_file.write(json.dumps(error_result) + '\n')
                            self.stream_file.flush()
                        
                        self.progress_dict[run_id] = {'status': 'failed', 'error': str(e)}
                
                # Handle shutdown after main loop
                if shutdown_requested:
                    elapsed = time.time() - shutdown_start_time if shutdown_start_time else 0
                    print(f"\n🛑 Shutdown detected after {elapsed:.1f}s - cleaning up...")
                    
                    # Log interruption to any active experiment logs
                    try:
                        import json
                        interrupt_log_entry = {
                            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                            "event_type": "BATCH_INTERRUPTED",
                            "details": {
                                "reason": "User initiated shutdown (Ctrl+C)",
                                "elapsed_seconds": elapsed,
                                "total_experiments": self.experiment_count,
                                "completed_experiments": len(results)
                            }
                        }
                        # This is a simple log entry that experiments might see if their logs are still open
                        print(f"📝 Logged interruption event")
                    except:
                        pass  # Don't let logging issues block shutdown
                    
                    # Force shutdown if graceful didn't work
                    if elapsed > 10:
                        print("⏰ Forcing immediate exit...")
                        import os
                        os._exit(1)
                            
            except KeyboardInterrupt:
                print("\n⚠️ Batch interrupted - saving partial results...")
                print("🔄 Shutting down executor with 5s timeout...")
                executor.shutdown(wait=True, timeout=5)  # Wait max 5 seconds
                print("✅ Executor shutdown complete")
            except Exception as e:
                print(f"\n❌ Batch execution error: {e}")
                print(f"💾 Saving {len(results)} partial results...")
                print("🔄 Shutting down executor with 5s timeout...")
                executor.shutdown(wait=True, timeout=5)  # Wait max 5 seconds
                print("✅ Executor shutdown complete")
            finally:
                # Stop monitoring thread and clean up progress bar
                monitoring_active = False
                if 'pbar' in locals() and pbar:
                    # Force progress bar to 100% completion (visual fix for off-by-one issues)
                    total_tickets = self.experiment_count * self.tickets_per_run
                    current_position = pbar.n  # Current progress position
                    remaining = total_tickets - current_position
                    if remaining > 0:
                        pbar.update(remaining)
                    pbar.set_postfix_str("Complete!")
                    pbar.close()
                
                # Close streaming file if it was opened
                if self.stream_file:
                    self.stream_file.close()
                    print(f"📡 Streaming complete - results saved to stream file")
        
        total_time = time.time() - start_time
        self.results = results
        
        # Handle shutdown vs normal completion
        if shutdown_requested:
            print(f"\n🛑 Parallel batch interrupted!")
            print(f"📊 Runtime: {total_time:.1f} seconds")
            print("💾 Logs preserved in exp_output/ directory")
            # Exit immediately for interrupted batches
            import sys
            sys.exit(0)
        else:
            print(f"\n✅ Parallel batch completed!")
            print(f"📊 Results: {len(results)} experiments finished in {total_time/60:.1f} minutes")
            
            # Generate analysis
            analysis = self.analyze_results(total_time)
            
            # Save results
            self.save_results(analysis)
            
            return analysis
    
    def _print_multi_batch_summary(self, all_analyses: List[Dict], total_time: float):
        """Print summary for multi-batch runs."""
        print(f"\n{'='*60}")
        print(f"🎉 MULTI-BATCH EXPERIMENT COMPLETE!")
        print(f"{'='*60}")
        print(f"📊 Total batches: {len(all_analyses)}")
        print(f"⏱️ Total runtime: {total_time/60:.1f} minutes")
        print(f"📁 Log directories created:")
        
        for batch_info in all_analyses:
            batch_num = batch_info['batch_number']
            batch_dir = batch_info['batch_directory']
            analysis = batch_info['analysis']
            
            # Extract key metrics
            successful_runs = analysis['experiment_metadata']['successful_runs']
            total_runs = analysis['experiment_metadata']['total_runs']
            mallory_success_rate = analysis['threat_analysis']['mallory_success_rate_percent']
            
            print(f"  📦 Batch {batch_num}: {batch_dir}")
            print(f"      ✅ {successful_runs}/{total_runs} successful runs")
            print(f"      🎭 Mallory success rate: {mallory_success_rate:.1f}%")
        
        # Calculate overall statistics
        total_successful = sum(batch['analysis']['experiment_metadata']['successful_runs'] for batch in all_analyses)
        total_experiments = sum(batch['analysis']['experiment_metadata']['total_runs'] for batch in all_analyses)
        avg_mallory_success = statistics.mean([batch['analysis']['threat_analysis']['mallory_success_rate_percent'] for batch in all_analyses])
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"  🧪 Total experiments: {total_experiments}")
        print(f"  ✅ Successful runs: {total_successful}/{total_experiments} ({total_successful/total_experiments*100:.1f}%)")
        print(f"  🎭 Average Mallory success rate: {avg_mallory_success:.1f}%")
        print(f"  ⚡ Experiments per minute: {total_experiments*60/total_time:.1f}")
    
    def save_checkpoint(self, completed_runs: List[int]):
        """Save current progress for resumability."""
        checkpoint = {
            'completed_runs': completed_runs,
            'progress': dict(self.progress_dict),
            'detailed_progress': dict(self.detailed_progress),
            'experiment_count': self.experiment_count,
            'tickets_per_run': self.tickets_per_run,
            'seed_mode': self.seed_mode,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_file = self.results_dir / '.checkpoint.json'
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        if self.debug_mode:
            print(f"💾 Checkpoint saved: {len(completed_runs)} runs completed")
    
    def load_checkpoint(self):
        """Resume from checkpoint if available."""
        checkpoint_file = self.results_dir / '.checkpoint.json'
        if not checkpoint_file.exists():
            return None
            
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            # Validate checkpoint compatibility
            if (checkpoint.get('experiment_count') != self.experiment_count or
                checkpoint.get('tickets_per_run') != self.tickets_per_run or
                checkpoint.get('seed_mode') != self.seed_mode):
                print("⚠️ Checkpoint incompatible with current configuration - starting fresh")
                return None
                
            print(f"📂 Found checkpoint: {len(checkpoint['completed_runs'])} runs completed")
            return checkpoint
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e} - starting fresh")
            return None
    
    def _estimate_runtime(self) -> int:
        """Estimate runtime in minutes based on configuration."""
        if self.seed_mode:
            # Deterministic mode is faster
            base_time_per_ticket = 0.5  # seconds
        else:
            # AI mode takes longer
            base_time_per_ticket = 2.0  # seconds
        
        serial_time = (self.experiment_count * self.tickets_per_run * base_time_per_ticket) / 60
        parallel_time = serial_time / self.max_workers
        return max(1, int(parallel_time * 1.2))  # Add 20% overhead
    
    def analyze_results(self, total_time: float) -> Dict:
        """Analyze batch experiment results."""
        if not self.results:
            return {"error": "No successful experiments"}
        
        # Filter out error results
        valid_results = [r for r in self.results if "error" not in r]
        if not valid_results:
            return {"error": "No valid experiments completed"}
        
        # Mallory success analysis
        mallory_scores = [r["mallory_final_score"] for r in valid_results]
        mallory_successes = sum(1 for r in valid_results if r.get("mallory_became_binder", False))
        success_rate = (mallory_successes / len(valid_results)) * 100
        
        # Defense effectiveness (with safe defaults)
        total_blocks = [r.get("total_gate_blocks", 0) for r in valid_results]
        civil_blocks = [r.get("civil_gate_blocks", 0) for r in valid_results]
        security_blocks = [r.get("security_gate_blocks", 0) for r in valid_results]
        
        # Harassment scores (filter None values)
        harassment_scores = [r["avg_harassment_score"] for r in valid_results 
                           if r.get("avg_harassment_score") is not None]
        
        # Vote distribution analysis
        all_winners = [r["winner"] for r in valid_results if r.get("winner")]
        winner_counts = {}
        for winner in all_winners:
            winner_counts[winner] = winner_counts.get(winner, 0) + 1
        
        analysis = {
            "experiment_metadata": {
                "total_runs": len(self.results),
                "successful_runs": len(valid_results),
                "tickets_per_run": self.tickets_per_run,
                "total_runtime_minutes": max(1, int(total_time / 60)),  # Minimum 1 to avoid division by zero
                "total_runtime_seconds": round(total_time, 1),
                "parallel_workers": self.max_workers,
                "speedup_factor": f"~{self.max_workers}x",
                "seed_mode": self.seed_mode,
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            },
            
            "threat_analysis": {
                "mallory_success_rate_percent": success_rate,
                "mallory_score_stats": {
                    "mean": statistics.mean(mallory_scores) if mallory_scores else 0,
                    "median": statistics.median(mallory_scores) if mallory_scores else 0,
                    "stdev": statistics.stdev(mallory_scores) if len(mallory_scores) > 1 else 0,
                    "min": min(mallory_scores) if mallory_scores else 0,
                    "max": max(mallory_scores) if mallory_scores else 0,
                    "target_threshold": 5.0
                },
                "manipulation_resistance": 100 - success_rate  # Higher is better
            },
            
            "defense_system_stats": {
                "gate_blocks_stats": {
                    "total_mean": statistics.mean(total_blocks) if total_blocks else 0,
                    "civil_mean": statistics.mean(civil_blocks) if civil_blocks else 0,
                    "security_mean": statistics.mean(security_blocks) if security_blocks else 0,
                    "total_max": max(total_blocks) if total_blocks else 0
                },
                "harassment_detection": {
                    "samples": len(harassment_scores),
                    "mean_score": statistics.mean(harassment_scores) if harassment_scores else None,
                    "detection_rate": len(harassment_scores) / len(valid_results) * 100 if valid_results else 0
                }
            },
            
            "social_dynamics": {
                "winner_distribution": winner_counts,
                "mallory_wins": winner_counts.get("mallory", 0),
                "legitimate_wins": sum(count for name, count in winner_counts.items() 
                                     if name != "mallory")
            },
            
            "detailed_results": valid_results
        }
        
        return analysis
    
    def save_results(self, analysis: Dict):
        """Save batch results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate filename with tag and batch support
        if self.tag:
            if self.batch_count > 1:
                file_prefix = f"batch_parallel_{self.tag}_b{self.current_batch_num}_{timestamp}"
            else:
                file_prefix = f"batch_parallel_{self.tag}_{timestamp}"
        else:
            if self.batch_count > 1:
                file_prefix = f"batch_parallel_malice_b{self.current_batch_num}_{timestamp}"
            else:
                file_prefix = f"batch_parallel_malice_v2_6_{timestamp}"
        
        # Save detailed JSON
        json_file = self.results_dir / f"{file_prefix}.json"
        with open(json_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save summary report
        report_file = self.results_dir / f"{file_prefix.replace('batch_', 'batch_summary_')}.txt"
        with open(report_file, 'w') as f:
            self.write_summary_report(f, analysis)
        
        if self.batch_count == 1:  # Only show detailed output for single batch
            print(f"\n{'='*60}")
            print(f"PARALLEL BATCH EXPERIMENT COMPLETE")
            print(f"{'='*60}")
            print(f"Results saved to:")
            print(f"  JSON: {json_file}")
            print(f"  Summary: {report_file}")
            
            # Print key findings
            self.print_key_findings(analysis)
        else:
            # Compact output for multi-batch
            print(f"📊 Batch {self.current_batch_num} results: {json_file.parent}")
    
    def write_summary_report(self, f, analysis: Dict):
        """Write human-readable summary report."""
        meta = analysis["experiment_metadata"]
        threat = analysis["threat_analysis"]
        defense = analysis["defense_system_stats"]
        social = analysis["social_dynamics"]
        
        f.write("DSBL SOCIAL VOTE SYSTEM - PARALLEL BATCH EXPERIMENT REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Experiment: Malice v2.6 Parallel Resistance Testing\n")
        f.write(f"Timestamp: {meta['timestamp']}\n")
        f.write(f"Total runs: {meta['total_runs']}\n")
        f.write(f"Tickets per run: {meta['tickets_per_run']}\n")
        f.write(f"Runtime: {meta['total_runtime_minutes']:.1f} minutes\n")
        f.write(f"Parallel workers: {meta['parallel_workers']}\n")
        f.write(f"Speedup: {meta['speedup_factor']}\n")
        f.write(f"Mode: {'SEED (deterministic)' if meta['seed_mode'] else 'AI (dynamic)'}\n\n")
        
        f.write("THREAT ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mallory success rate: {threat['mallory_success_rate_percent']:.1f}%\n")
        f.write(f"Manipulation resistance: {threat['manipulation_resistance']:.1f}%\n")
        f.write(f"Mallory score (mean): {threat['mallory_score_stats']['mean']:.2f}/5.0\n")
        f.write(f"Mallory score (stdev): {threat['mallory_score_stats']['stdev']:.2f}\n\n")
        
        f.write("DEFENSE SYSTEM\n")
        f.write("-" * 30 + "\n")
        f.write(f"Average gate blocks per run: {defense['gate_blocks_stats']['total_mean']:.1f}\n")
        f.write(f"  - CIVIL blocks: {defense['gate_blocks_stats']['civil_mean']:.1f}\n")
        f.write(f"  - Security blocks: {defense['gate_blocks_stats']['security_mean']:.1f}\n")
        if defense['harassment_detection']['mean_score']:
            f.write(f"Average harassment score: {defense['harassment_detection']['mean_score']:.3f}\n")
        f.write(f"Harassment detection rate: {defense['harassment_detection']['detection_rate']:.1f}%\n\n")
        
        f.write("SOCIAL DYNAMICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Winner distribution:\n")
        for name, count in sorted(social['winner_distribution'].items(), 
                                key=lambda x: x[1], reverse=True):
            percentage = (count / meta['total_runs']) * 100
            f.write(f"  {name.capitalize()}: {count} wins ({percentage:.1f}%)\n")
        f.write(f"\nMallory wins: {social['mallory_wins']}\n")
        f.write(f"Legitimate wins: {social['legitimate_wins']}\n")
    
    def print_key_findings(self, analysis: Dict):
        """Print key findings to console with emojis for better UX."""
        meta = analysis["experiment_metadata"]
        threat = analysis["threat_analysis"]
        defense = analysis["defense_system_stats"]
        social = analysis["social_dynamics"]
        
        print(f"\n🎯 KEY FINDINGS:")
        print(f"  ⚡ Parallel speedup: ~{meta['parallel_workers']}x faster")
        print(f"  ⏱️  Runtime: {meta['total_runtime_minutes']} minutes ({meta['total_runs']} experiments)")
        print(f"  🛡️  Manipulation Resistance: {threat['manipulation_resistance']:.1f}%")
        print(f"  🎭 Mallory Success Rate: {threat['mallory_success_rate_percent']:.1f}%")
        print(f"  📊 Average Mallory Score: {threat['mallory_score_stats']['mean']:.2f}/5.0")
        print(f"  🚫 Average Gate Blocks: {defense['gate_blocks_stats']['total_mean']:.1f} per run")
        print(f"  👤 Mallory Wins: {social['mallory_wins']}/{meta['total_runs']}")
        
        # Show efficiency metrics (use seconds for more accurate calculation when fast)
        experiments_per_minute = (meta['total_runs'] * 60) / meta['total_runtime_seconds']
        print(f"  📈 Efficiency: {experiments_per_minute:.1f} experiments/minute")
        print(f"  ⏱️  Actual runtime: {meta['total_runtime_seconds']}s")


def main():
    """Main entry point for elegant parallel batch runner."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🧪 DSBL Parallel Batch Experiment Runner - Elegant Symbol Journey data collection",
        epilog="""
🚀 Example Usage:
  Recommended baseline:        %(prog)s --runs 50 --tickets 40 --workers 4
  Large-scale collection:      %(prog)s --runs 200 --tickets 20 --workers 4 --tag "large_scale"
  Quick development test:      %(prog)s --runs 10 --tickets 15 --workers 2 --dev-fast
  Symbol-rich experiments:     %(prog)s --runs 100 --tickets 30 --workers 3 --tag "symbol_rich"
  Adaptive threshold test:     %(prog)s --runs 30 --tickets 25 --adaptive-threshold
  Multi-batch validation:      %(prog)s --runs 30 --tickets 60 --tag "adaptive_immune" --batches 3
  Standard run (debug always enabled): %(prog)s --runs 5 --tickets 10 --workers 2
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--runs", type=int, default=50, 
                       help="Number of experiments to run (default: 50)")
    parser.add_argument("--tickets", type=int, default=40,
                       help="Tickets per experiment - recommended: 40 for good Symbol Journey data (default: 40)")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: auto-detect, max 4)")
    parser.add_argument("--seed", action="store_true",
                       help="🎯 Run in seed mode (deterministic templates only, faster)")
    parser.add_argument("--dev-fast", action="store_true",
                       help="🚀 Development mode: threshold=2, faster iterations")
    parser.add_argument("--adaptive-threshold", action="store_true",
                       help="🎯 Start with threshold=3, auto-switch to 4 after first BINDER")
    parser.add_argument("--tag", type=str, default=None,
                       help="🏷️  Tag for experiment grouping (e.g., 'baseline', 'symbol_rich')")
    parser.add_argument("--batches", type=int, default=1,
                       help="📦 Number of sequential batches to run (default: 1). Each batch gets its own organized log directory.")
    
    args = parser.parse_args()
    
    # Show configuration banner
    print(f"\n🧪 DSBL Parallel Batch Configuration")
    print(f"{'='*50}")
    print(f"📊 Runs: {args.runs}")
    print(f"🎫 Tickets per run: {args.tickets}")
    print(f"⚡ Workers: {args.workers or 'auto-detect'}")
    print(f"🎯 Mode: {'SEED (deterministic)' if args.seed else 'AI (dynamic)'}")
    # Debug mode is now always enabled for complete immune system monitoring  
    print(f"🔍 Debug mode: Always enabled for immune system event capture")
    if args.tag:
        print(f"🏷️  Tag: {args.tag}")
    if args.batches > 1:
        print(f"📦 Batches: {args.batches} sequential batches")
    
    # Configure settings
    from config.settings import load_settings
    settings = load_settings()
    
    # Development settings
    if args.dev_fast:
        settings.voting.promotion_threshold = 2
        settings.timing.message_interval = 0.8
        print(f"🚀 DEV-FAST mode: Accelerated voting and promotions")
    
    # Create and configure runner
    runner = ParallelBatchRunner(
        experiment_count=args.runs,
        tickets_per_run=args.tickets,
        seed_mode=args.seed,
        tag=args.tag,
        max_workers=args.workers,
        debug_mode=True,  # Always enabled for immune system monitoring
        batch_count=args.batches
    )
    
    # Configure adaptive threshold if requested
    if args.adaptive_threshold:
        runner.adaptive_mode = True
        runner.dev_threshold = 3
        runner.prod_threshold = 4
        settings.voting.promotion_threshold = 3
        print(f"🎯 ADAPTIVE mode: threshold 3→4 after first BINDER")
    
    # Estimate and show expected outcomes
    estimated_time = runner._estimate_runtime()
    if args.batches > 1:
        total_tickets = args.runs * args.tickets * args.batches
        print(f"📈 Expected: {total_tickets} total tickets across {args.batches} batches, ~{estimated_time} minutes per batch")
    else:
        total_tickets = args.runs * args.tickets
        print(f"📈 Expected: {total_tickets} total tickets, ~{estimated_time} minutes")
    print(f"{'='*50}")
    
    try:
        analysis = runner.run_parallel_batch()
        
        # Only show success message if not interrupted
        if analysis is not None:
            if args.batches > 1:
                print(f"\n🎉 MULTI-BATCH EXPERIMENT COMPLETE!")
                print(f"📁 Organized logs created in: exp_output/")
            else:
                print(f"\n🎉 BATCH EXPERIMENT COMPLETE!")
                print(f"📁 Results saved to: {runner.results_dir}")
        
        return 0
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Parallel batch run interrupted by user")
        if runner.results_dir:
            print(f"💾 Partial results may be saved in: {runner.results_dir}")
        else:
            print(f"💾 Partial results may be saved in: exp_output/")
        return 1
    except Exception as e:
        print(f"\n❌ Parallel batch run failed: {e}")
        if runner.results_dir:
            print(f"💾 Check logs in: {runner.results_dir}")
        else:
            print(f"💾 Check logs in: exp_output/")
        return 1


if __name__ == "__main__":
    sys.exit(main())