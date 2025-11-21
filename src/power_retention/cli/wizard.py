import inquirer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich import print as rprint
import sys
import os

console = Console()

def run_wizard():
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]Power Retention MLX[/bold cyan]\n"
        "[yellow]Interactive Setup Wizard[/yellow]",
        border_style="blue"
    ))

    questions = [
        inquirer.List('action',
                      message="What would you like to do?",
                      choices=[
                          ('Train a new model', 'train'),
                          ('Run inference (Chat)', 'chat'),
                          ('Launch Dashboard', 'dashboard'),
                          ('Exit', 'exit')
                      ],
                      ),
    ]
    
    answers = inquirer.prompt(questions)
    
    if not answers:
        return

    action = answers['action']

    if action == 'train':
        setup_training()
    elif action == 'chat':
        setup_inference()
    elif action == 'dashboard':
        launch_dashboard()
    elif action == 'exit':
        sys.exit(0)

def setup_training():
    console.print("\n[bold green]Training Setup[/bold green]")
    
    questions = [
        inquirer.List('model_size',
                      message="Select model size",
                      choices=[
                          ('Ultra-Tiny (2M params, <8GB RAM)', 'ultra-tiny'),
                          ('Tiny (17M params, 8GB RAM)', 'tiny'),
                          ('Small (125M params, 16GB RAM)', 'small'),
                          ('Custom', 'custom')
                      ],
                      ),
        inquirer.List('data_source',
                      message="Select data source",
                      choices=[
                          ('Synthetic (No download needed)', 'synthetic'),
                          ('HuggingFace Dataset (Requires internet)', 'real')
                      ],
                      ),
        inquirer.Text('steps',
                      message="Number of training steps",
                      default="100"
                      ),
    ]
    
    answers = inquirer.prompt(questions)
    if not answers: return

    cmd = f"python3 train.py --mode llm --model {answers['model_size']} --steps {answers['steps']}"
    
    if answers['data_source'] == 'real':
        cmd += " --real-data"
        
    console.print(f"\n[bold]Ready to run:[/bold]\n[cyan]{cmd}[/cyan]")
    
    if inquirer.confirm("Start training now?", default=True):
        os.system(cmd)

def setup_inference():
    console.print("\n[bold green]Inference Setup[/bold green]")
    # TODO: Scan for checkpoints
    console.print("[yellow]Scanning for checkpoints...[/yellow]")
    
    # Placeholder for checkpoint selection
    cmd = "python3 train.py --quick-test" # Just a placeholder for now
    
    console.print("To run inference, we need a trained model. For now, running quick test.")
    if inquirer.confirm("Run quick test?", default=True):
        os.system(cmd)

def launch_dashboard():
    console.print("\n[bold green]Launching Dashboard...[/bold green]")
    cmd = "python3 -m power_retention.dashboard.app"
    console.print(f"Running: {cmd}")
    # os.system(cmd) # Uncomment when dashboard is ready
    console.print("[red]Dashboard not implemented yet![/red]")

if __name__ == "__main__":
    run_wizard()
