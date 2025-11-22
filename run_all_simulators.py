#!/usr/bin/env python3
"""
Distributed Systems Simulator Launcher
Run this to see menu and launch simulators
"""

import sys
import subprocess
import os

SIMULATORS = {
    '1': {
        'name': 'Metastable Failure Simulator',
        'file': 'metastable_simulator.py',
        'description': 'Demonstrates retry storms and congestive collapse',
        'concepts': ['Metastability', 'Retry storms', 'Coordination overhead']
    },
    '2': {
        'name': 'CAP/PACELC Consistency Simulator',
        'file': 'cap_consistency_simulator.py',
        'description': 'Explores consistency vs availability vs latency tradeoffs',
        'concepts': ['CAP theorem', 'PACELC', 'Quorum systems', 'Strong consistency']
    },
    '3': {
        'name': 'CRDT Simulator',
        'file': 'crdt_simulator.py',
        'description': 'Shows how CRDTs achieve eventual consistency without coordination',
        'concepts': ['CRDTs', 'Eventual consistency', 'Convergence', 'Gossip protocols']
    },
    '4': {
        'name': 'Cross-System Interaction (CSI) Failure Simulator',
        'file': 'csi_failure_simulator.py',
        'description': 'Demonstrates how correct systems fail through interactions',
        'concepts': ['CSI failures', 'Schema conflicts', 'Config incoherence', 'API violations']
    }
}

def print_banner():
    """Print welcome banner"""
    print("=" * 70)
    print("  🔬 DISTRIBUTED SYSTEMS FAILURE & CONSISTENCY SIMULATOR 🔬")
    print("=" * 70)
    print()
    print("Interactive simulations for understanding distributed systems:")
    print("  • Metastable failures and retry storms")
    print("  • CAP theorem and consistency tradeoffs")
    print("  • CRDTs and eventual consistency")
    print("  • Cross-system interaction failures")
    print()
    print("=" * 70)
    print()

def print_menu():
    """Print simulator menu"""
    print("Available Simulators:")
    print()
    
    for key, sim in SIMULATORS.items():
        print(f"  [{key}] {sim['name']}")
        print(f"      {sim['description']}")
        print(f"      Concepts: {', '.join(sim['concepts'])}")
        print()
    
    print("  [a] Run all simulators (in separate terminals)")
    print("  [h] Show help and requirements")
    print("  [q] Quit")
    print()

def check_requirements():
    """Check if required packages are installed"""
    required = ['streamlit', 'simpy', 'plotly', 'numpy', 'pandas']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("⚠️  Missing required packages:")
        print()
        for pkg in missing:
            print(f"   • {pkg}")
        print()
        print("Install with:")
        print(f"   pip install {' '.join(missing)}")
        print()
        return False
    
    return True

def show_help():
    """Show help information"""
    print()
    print("=" * 70)
    print("HELP & REQUIREMENTS")
    print("=" * 70)
    print()
    print("Requirements:")
    print("  • Python 3.7+")
    print("  • streamlit (web UI)")
    print("  • simpy (discrete event simulation)")
    print("  • plotly (interactive plots)")
    print("  • numpy (numerical computing)")
    print("  • pandas (data analysis)")
    print()
    print("Installation:")
    print("  pip install streamlit simpy plotly numpy pandas")
    print()
    print("Running a simulator:")
    print("  streamlit run <simulator_file>.py")
    print()
    print("Controls:")
    print("  • Use sliders to adjust parameters")
    print("  • Click 'Run Simulation' to start")
    print("  • Explore expandable sections for theory")
    print("  • Try different scenarios and observe results")
    print()
    print("Tips:")
    print("  • Start with default parameters to understand baseline")
    print("  • Change one parameter at a time to see its effect")
    print("  • Read the educational content in expandable sections")
    print("  • Try the suggested experiments")
    print()
    print("=" * 70)
    print()

def run_simulator(sim_file):
    """Run a simulator using streamlit"""
    if not os.path.exists(sim_file):
        print(f"❌ Error: {sim_file} not found!")
        print("   Make sure you're in the correct directory.")
        return False
    
    print(f"🚀 Launching {sim_file}...")
    print("   (Press Ctrl+C to stop)")
    print()
    
    try:
        subprocess.run(['streamlit', 'run', sim_file])
        return True
    except KeyboardInterrupt:
        print("\n⏹️  Stopped simulator")
        return True
    except FileNotFoundError:
        print("❌ Error: streamlit not found!")
        print("   Install with: pip install streamlit")
        return False
    except Exception as e:
        print(f"❌ Error running simulator: {e}")
        return False

def run_all_simulators():
    """Attempt to run all simulators"""
    print("🚀 Launching all simulators...")
    print()
    print("Note: This will open multiple browser tabs.")
    print("      Close tabs to stop individual simulators.")
    print()
    
    for key, sim in SIMULATORS.items():
        print(f"  Starting: {sim['name']}")
        if os.name == 'nt':  # Windows
            subprocess.Popen(['start', 'cmd', '/k', 'streamlit', 'run', sim['file']], shell=True)
        else:  # Unix/Linux/Mac
            subprocess.Popen(['streamlit', 'run', sim['file']])
    
    print()
    print("✅ All simulators launched!")
    print("   Check your browser for tabs.")

def main():
    """Main launcher loop"""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        print("Please install required packages first.")
        sys.exit(1)
    
    while True:
        print_menu()
        choice = input("Select option: ").strip().lower()
        print()
        
        if choice == 'q':
            print("👋 Goodbye!")
            break
        
        elif choice == 'h':
            show_help()
        
        elif choice == 'a':
            run_all_simulators()
            input("\nPress Enter to continue...")
        
        elif choice in SIMULATORS:
            sim = SIMULATORS[choice]
            print(f"Running: {sim['name']}")
            print(f"File: {sim['file']}")
            print()
            run_simulator(sim['file'])
            print()
            input("Press Enter to return to menu...")
        
        else:
            print(f"❌ Invalid choice: {choice}")
            print()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
        sys.exit(0)
