#!/usr/bin/env python3
"""
Multi-Agent Adaptive Immune System - Interactive Interface
=========================================================

Interactive CLI for running Multi-Agent Adaptive Immune System.
Simplified interface with essential functionality only.
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
    print("DSBL MULTI-AGENT ADAPTIVE IMMUNE SYSTEM")
    print("=" * 60)
    print()
    print("Multi-Agent Adaptive Immune System Research Platform")
    print("Interactive social dynamics with emergent behaviors")
    print()
    print("Available Options:")
    print()
    print("1. Run Interactive Experiment")
    print("   Multi-Agent Adaptive Immune System")
    print("   - 7 AI agents with coordinated behaviors")
    print("   - Adaptive immune system with real-time adjustments")
    print("   - Advanced gate processing and social dynamics")
    print("   - Requires OpenAI API key")
    print()
    print("2. Help & Documentation")
    print("   Learn about the system and controls")
    print()
    print("0. Exit")
    print()
    print("-" * 60)

def run_interactive_experiment():
    """Run the interactive multi-agent experiment."""
    print("\\nStarting Multi-Agent Adaptive Immune System...")
    print("   Interactive experiment with real-time coordination")
    print()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  WARNING: No OPENAI_API_KEY found in environment")
        print("   AI agents will fall back to template mode")
        print()
        choice = input("Continue anyway? (y/N): ").strip().lower()
        if choice != 'y':
            return
    
    try:
        # Create organized log directory for interactive experiments
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        tag = f"interactive_{timestamp}"
        log_dir = f"exp_output/{tag}"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create subdirectories to match batch structure
        os.makedirs(f"{log_dir}/events", exist_ok=True)    # Main event logs
        os.makedirs(f"{log_dir}/metrics", exist_ok=True)   # System metrics
        
        print(f"📁 Logs will be saved to: {log_dir}/")
        print()
        
        from experiments.malice import MaliceExperiment
        # Create experiment with organized logging
        experiment = MaliceExperiment(seed_mode=False)
        
        # Update audit logger to use organized directory structure
        original_audit_file = experiment.vote_system.audit_logger.audit_file
        filename = os.path.basename(original_audit_file)
        experiment.vote_system.audit_logger.audit_file = f"{log_dir}/events/{filename}"
        
        # Update metrics file to use organized metrics directory
        if experiment.vote_system.audit_logger.debug_file:
            debug_filename = os.path.basename(experiment.vote_system.audit_logger.debug_file)
            experiment.vote_system.audit_logger.debug_file = f"{log_dir}/metrics/{debug_filename}"
        
        experiment.run_interactive_mode()
        
        print(f"\\n📊 Experiment complete! Results saved to: {log_dir}/")
        
    except Exception as e:
        print(f"❌ Error starting malice experiment: {e}")
        print("\\nTry running in seed mode or check your setup.")







def show_help():
    """Show help and documentation."""
    print("\\nMulti-Agent Adaptive Immune System Help")
    print("=" * 50)
    print()
    print("System Overview:")
    print("   • Multi-Agent Adaptive Immune System")
    print("   • Real-time coordination between Eve, Dave, and Zara")
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
    print("   • OPENAI_API_KEY=your_api_key_here")
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
            
            if choice == "0":
                print("Bye!")
                break
                
            elif choice == "1":
                run_interactive_experiment()
                
            elif choice == "2":
                show_help()
                
            else:
                print("\\n❌ Invalid choice. Please enter 0-2.")
            
            if choice != "0":
                input("\\nPress Enter to return to main menu...")
                
        except KeyboardInterrupt:
            print("\\n\\nGoodbye!")
            break
        except Exception as e:
            print(f"\\n❌ Unexpected error: {e}")
            print("Please try again or report this issue.")

if __name__ == "__main__":
    main()