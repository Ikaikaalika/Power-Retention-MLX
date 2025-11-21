import argparse
import sys
from power_retention.cli.wizard import run_wizard

def main():
    parser = argparse.ArgumentParser(description="Power Retention MLX CLI")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Wizard command
    subparsers.add_parser('wizard', help='Run interactive setup wizard')
    
    # Dashboard command
    subparsers.add_parser('dashboard', help='Launch web dashboard')

    args = parser.parse_args()

    if args.command == 'wizard' or args.command is None:
        run_wizard()
    elif args.command == 'dashboard':
        print("Launching dashboard... (Not implemented yet)")
        # TODO: Import and run dashboard
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
