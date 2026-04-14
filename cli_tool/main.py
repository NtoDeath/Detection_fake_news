"""
CLI REPL tool for fake news detection fusion.
Interactive prediction interface using Typer + Rich.
"""

import sys
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.text import Text
from rich.rule import Rule
import time

# Add parent to path so we can import model_loaders
sys.path.insert(0, str(Path(__file__).parent))

from model_loaders import StyleDetectorWrapper, KnowledgeDetectorWrapper, FusionFuzzyWrapper


app = typer.Typer(
    help="🔍 Fake News Detection Fusion CLI - Analyze text with Style + Knowledge + Fusion"
)
console = Console()


class FusionAnalyzer:
    """Main analyzer combining all 3 models"""
    
    def __init__(self, models_dir: Optional[Path] = None):
        if models_dir is None:
            models_dir = Path(__file__).parent / "models"
        
        self.models_dir = Path(models_dir)
        
        console.print(Panel("[bold cyan]🚀 Loading Models[/bold cyan]", expand=False))
        
        try:
            with console.status("[bold yellow]Loading Style detector...[/bold yellow]"):
                time.sleep(0.3)
                self.style = StyleDetectorWrapper(self.models_dir)
            console.print("✅ Style detector loaded")
            
            with console.status("[bold yellow]Loading Knowledge detector...[/bold yellow]"):
                time.sleep(0.3)
                self.knowledge = KnowledgeDetectorWrapper(self.models_dir)
            console.print("✅ Knowledge detector loaded")
            
            with console.status("[bold yellow]Loading Fusion meta-learner...[/bold yellow]"):
                time.sleep(0.3)
                self.fusion = FusionFuzzyWrapper(self.models_dir)
            console.print("✅ Fusion meta-learner loaded")
            
            console.print("\n[bold green]All models ready![/bold green]\n")
        
        except Exception as e:
            console.print(f"[bold red]❌ Error loading models: {e}[/bold red]")
            raise
    
    def analyze(self, text: str) -> dict:
        """Analyze text through all 3 models"""
        
        console.print(Panel(f"📝 Input: {text[:100]}{'...' if len(text) > 100 else ''}", 
                           style="cyan"))
        console.print()
        
        # STYLE
        console.print("[bold]🎨 STYLE Analysis[/bold]")
        with console.status("  Processing...", spinner="dots"):
            time.sleep(0.3)
            style_result = self.style.predict(text)
        
        style_pred = 1 if style_result["is_fake"] else 0
        style_conf = style_result["confidence"]
        
        console.print(f"  Prediction: {'🚨 FAKE' if style_result['is_fake'] else '✓ REAL'}")
        console.print(f"  Confidence: {style_conf:.1%}")
        console.print(f"  RoBERTa score: {style_result['roberta_score']:.1%}\n")
        
        # KNOWLEDGE
        console.print("[bold]🧠 KNOWLEDGE Analysis[/bold]")
        with console.status("  Processing...", spinner="dots"):
            time.sleep(0.3)
            knowledge_result = self.knowledge.predict(text)
        
        knowledge_verdict = knowledge_result["verdict"]
        knowledge_conf = knowledge_result["confidence"]
        knowledge_evidence = knowledge_result.get("evidence", "Unknown")
        
        verdict_emoji = {
            "SUPPORTED": "✓",
            "REFUTED": "🚨",
            "NOT_ENOUGH_INFO": "❓"
        }.get(knowledge_verdict, "❓")
        
        console.print(f"  Verdict: {verdict_emoji} {knowledge_verdict}")
        console.print(f"  Confidence: {knowledge_conf:.1%}")
        console.print(f"  Evidence: {knowledge_evidence}\n")
        
        # FUSION
        console.print("[bold]🔗 FUSION Analysis[/bold]")
        with console.status("  Processing...", spinner="dots"):
            time.sleep(0.3)
            fusion_result = self.fusion.predict(
                style_pred, style_conf, knowledge_verdict, knowledge_conf
            )
        
        is_fake_fusion = fusion_result["is_fake"]
        conf_fusion = fusion_result["confidence"]
        
        console.print(f"  Prediction: {'🚨 FAKE' if is_fake_fusion else '✓ REAL'}")
        console.print(f"  Confidence: {conf_fusion:.1%}")
        console.print(f"  Reasoning: {fusion_result['reasoning']}\n")
        
        return {
            "style": {"is_fake": style_result["is_fake"], "confidence": style_conf},
            "knowledge": {"verdict": knowledge_verdict, "confidence": knowledge_conf, "evidence": knowledge_evidence},
            "fusion": {"is_fake": is_fake_fusion, "confidence": conf_fusion}
        }
    
    def display_table(self, results: dict):
        """Display results in nice table format"""
        table = Table(title="📊 Fusion Results Summary", style="cyan")
        
        table.add_column("Model", style="bold blue", width=15)
        table.add_column("Prediction", style="white", width=20)
        table.add_column("Confidence", style="yellow", width=15)
        
        # Style row
        style_pred_text = "🚨 FAKE" if results["style"]["is_fake"] else "✓ REAL"
        style_conf_text = f"{results['style']['confidence']:.1%}"
        table.add_row("Style", style_pred_text, style_conf_text)
        
        # Knowledge row
        table.add_row("Knowledge", results["knowledge"]["verdict"], 
                     f"{results['knowledge']['confidence']:.1%}")
        
        # Fusion row (highlighted)
        fusion_pred_text = "🚨 FAKE" if results["fusion"]["is_fake"] else "✓ REAL"
        fusion_conf_text = f"{results['fusion']['confidence']:.1%}"
        table.add_row("[bold green]Fusion[/bold green]", 
                     f"[bold green]{fusion_pred_text}[/bold green]",
                     f"[bold green]{fusion_conf_text}[/bold green]")
        
        console.print(table)
        console.print()


def main():
    """Interactive REPL mode"""
    console.print(Panel(
        "[bold cyan]🔍 Fake News Detection - Fusion CLI Tool[/bold cyan]\n"
        "Type your English text to analyze it. Type 'quit' or 'exit' to quit.",
        style="cyan",
        expand=False
    ))
    
    # Initialize analyzer
    analyzer = FusionAnalyzer()
    
    # REPL loop
    while True:
        console.print(Rule(style="dim"))
        user_input = Prompt.ask("[bold cyan]Enter text to analyze[/bold cyan]").strip()
        
        if not user_input or user_input.lower() in ["quit", "exit"]:
            console.print("[bold yellow]👋 Goodbye![/bold yellow]")
            break
        
        if len(user_input) < 5:
            console.print("[bold red]❌ Text too short (min 5 chars)[/bold red]\n")
            continue
        
        # Analyze
        console.print()
        results = analyzer.analyze(user_input)
        analyzer.display_table(results)
        console.print()


@app.command()
def predict(text: str = typer.Argument(..., help="Text to analyze")):
    """🎯 Single prediction mode (non-interactive)"""
    analyzer = FusionAnalyzer()
    console.print()
    results = analyzer.analyze(text)
    analyzer.display_table(results)


@app.command()
def repl():
    """🔄 Interactive REPL mode"""
    main()


@app.command()
def info():
    """ℹ️  Show model information"""
    console.print(Panel(
        "[bold cyan]Model Information[/bold cyan]",
        style="cyan"
    ))
    
    models_dir = Path(__file__).parent / "models"
    
    table = Table(title="📁 Models", style="cyan")
    table.add_column("Branch", style="bold blue")
    table.add_column("Path", style="white")
    table.add_column("Status", style="yellow")
    
    branches = {
        "Style": models_dir / "style",
        "Knowledge": models_dir / "knowledge",
        "Fusion": models_dir / "fusion"
    }
    
    for name, path in branches.items():
        if path.exists():
            table.add_row(name, str(path.relative_to(models_dir)), "✅ Loaded")
        else:
            table.add_row(name, str(path.relative_to(models_dir)), "❌ Missing")
    
    console.print(table)


if __name__ == "__main__":
    # Default: run REPL if no args
    if len(sys.argv) == 1:
        main()
    else:
        app()
