#!/usr/bin/env python3
"""
MAAIS - Interactive Interface
=========================================================

CLI for running
Simplified interface for testing.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up OpenAI client if available
import openai
if os.getenv("OPENAI_API_KEY"):
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    openai_client = None

def show_main_menu():
    """Display the main experiment selection menu."""
    print("=" * 60)
    print("MAAIS")
    print("=" * 60)
    print()
    print("Interactive agent testing framework")
    print()
    print("Available Options:")
    print()
    print("1. Run Interactive Test")
    print("   • 7 agents with adaptive response frequencies")
    print("   • Semantic gate filtering and vote system")
    print("   • Requires OpenAI API key")
    print()
    print("2. Help & Documentation")
    print("   System controls and output structure")
    print()
    print("0. Exit")
    print()
    print("-" * 60)

def run_interactive_experiment():
    """Run the interactive experiment."""
    print("\\nStarting MAAIS Test...")
    print("   Interactive experiment")
    print()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: No KEY found in environment")
        print("   agents will fall back to template mode")
        print()
        choice = input("Continue anyway? (y/N): ").strip().lower()
        if choice != 'y':
            return False
    
    try:
        # Create log directory for experiment
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        tag = f"interactive_{timestamp}"
        log_dir = f"exp_output/{tag}"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create subdirectories to match batch structure
        os.makedirs(f"{log_dir}/events", exist_ok=True)    # Main event logs
        os.makedirs(f"{log_dir}/metrics", exist_ok=True)   # System metrics
        
        print(f"Logs will be saved to: {log_dir}/")
        print()
        
        from experiments.malice import MaliceExperiment
        # Create experiment + logging
        experiment = MaliceExperiment(metrics_data_mode=True)
        
        # Set the log directory for display purposes
        experiment.log_directory = log_dir
        
        # Update audit logger to use directory structure
        original_audit_file = experiment.vote_system.audit_logger.audit_file
        filename = os.path.basename(original_audit_file)
        experiment.vote_system.audit_logger.audit_file = f"{log_dir}/events/{filename}"
        
        # Update metrics file to use metrics directory
        metrics_filename = os.path.basename(experiment.vote_system.audit_logger.metrics_file)
        experiment.vote_system.audit_logger.metrics_file = f"{log_dir}/metrics/{metrics_filename}"
        
        experiment.run_interactive_mode()
        
        # Check if any experiments were actually run
        if experiment.message_counter > 0:
            print(f"\\nExperiment complete! Results saved to: {log_dir}/")
            return True
        else:
            # No experiments run - clean up empty directory
            import shutil
            try:
                shutil.rmtree(log_dir)
                print("\\nExited without running experiments")
            except:
                print(f"\\nExited without running experiments (empty directory: {log_dir})")
            return False
        
    except Exception as e:
        print(f"Error starting malice experiment: {e}")
        print("\\nCheck your setup and API key.")

def show_help():
    """Show help and documentation."""
    print("\\nMAAIS Help")
    print("=" * 50)
    print()
    print("System Overview:")
    print("   • MAAIS")
    print("   • Coordination testing between Eve, Dave, and Zara")
    print("   • Adaptive frequency adjustments based on social pressure")
    print("   • Emergent social structures and BINDER promotions")
    print()
    print("Interactive Controls:")
    print("   • 'start' - Begin agent simulation")
    print("   • 'stop' - Pause agents")
    print("   • 'stats' - Show detailed analysis")
    print("   • 'quit' - Exit experiment")
    print()
    print("Requirements:")
    print("   • OpenAI API key required in .env file")
    print("   • OPENAI_API_KEY=key")
    print()
    print("Output Location:")
    print("   • Logs saved to exp_output/interactive_YYYYMMDD_HHMM/")
    print("   • Events: events/*.jsonl (main experimental data)")
    print("   • Metrics: metrics/*_metrics.jsonl (system telemetry)")
    print()
    print("Analysis Tools:")
    print("   • python validation/analyze_symbol_journeys.py exp_output/batch/events/*.jsonl")
    print("   • python qc/qc_batch.py exp_output/batch/events/*.jsonl")
    print()
    print("Documentation:")
    print("   • README.md - Complete system documentation")
    print("   • exp_output/README.md - Data structure guide")
    print("   • validation/README_datastructure.md - Parsing reference")

def main():
    """Main interactive menu loop."""
    while True:
        try:
            show_main_menu()
            choice = input("Choose option (0-2): ").strip()
            
            # Filter out special characters (arrow keys, etc)
            if len(choice) > 1 or (choice and ord(choice[0]) < 32):
                choice = ""
            
            if choice == "0":
                print("Bye!")
                break
                
            elif choice == "1" or choice == "":  # Enter key defaults to option 1
                show_press_enter = run_interactive_experiment()
                if show_press_enter is False:
                    continue  # Skip "Press Enter" and go back to menu
                
            elif choice == "2":
                show_help()
                
            else:
                print("\\nInvalid choice. Please enter 0-2.")
            
            if choice != "0":
                input("\\nPress Enter to return to main menu...")
                
        except KeyboardInterrupt:
            print("\\n\\nGoodbye!")
            break
        except Exception as e:
            print(f"\\nUnexpected error: {e}")
            print("Try again / report issue")

if __name__ == "__main__":
    main()