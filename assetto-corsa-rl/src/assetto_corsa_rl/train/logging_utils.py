"""Colored logging utilities using ANSI colors and art library."""

import sys
from typing import Any, Optional


class Colors:
    """ANSI color codes."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def log_info(message: str, bold: bool = False) -> None:
    """Log info message in cyan."""
    prefix = f"{Colors.BOLD}{Colors.CYAN}[INFO]{Colors.END}"
    msg = f"{Colors.BOLD if bold else ''}{message}{Colors.END}"
    print(f"{prefix} {msg}", flush=True)


def log_success(message: str, bold: bool = True) -> None:
    """Log success message in green."""
    prefix = f"{Colors.BOLD}{Colors.GREEN}[✓]{Colors.END}"
    msg = f"{Colors.BOLD if bold else ''}{message}{Colors.END}"
    print(f"{prefix} {msg}", flush=True)


def log_warning(message: str, bold: bool = False) -> None:
    """Log warning message in yellow."""
    prefix = f"{Colors.BOLD}{Colors.YELLOW}[!]{Colors.END}"
    msg = f"{Colors.BOLD if bold else ''}{message}{Colors.END}"
    print(f"{prefix} {msg}", flush=True)


def log_error(message: str, bold: bool = True) -> None:
    """Log error message in red."""
    prefix = f"{Colors.BOLD}{Colors.RED}[✗]{Colors.END}"
    msg = f"{Colors.BOLD if bold else ''}{message}{Colors.END}"
    print(f"{prefix} {msg}", flush=True)


def log_metric(key: str, value: Any, precision: int = 4) -> None:
    """Log a metric with key-value formatting."""
    if isinstance(value, float):
        formatted_val = f"{value:.{precision}f}"
    else:
        formatted_val = str(value)
    prefix = f"{Colors.BOLD}{Colors.BLUE}[METRIC]{Colors.END}"
    print(
        f"{prefix} {Colors.BOLD}{key}{Colors.END}: {Colors.CYAN}{formatted_val}{Colors.END}",
        flush=True,
    )


def log_training_step(
    step: int,
    total_steps: int,
    loss: Optional[float] = None,
    lr: Optional[float] = None,
    epsilon: Optional[float] = None,
) -> None:
    """Log a training step with progress."""
    progress = f"{Colors.BOLD}{Colors.CYAN}[{step}/{total_steps}]{Colors.END}"
    info = f"{progress}"

    if loss is not None:
        info += f" {Colors.BOLD}loss{Colors.END}: {Colors.YELLOW}{loss:.6f}{Colors.END}"
    if lr is not None:
        info += f" {Colors.BOLD}lr{Colors.END}: {Colors.YELLOW}{lr:.2e}{Colors.END}"
    if epsilon is not None:
        info += f" {Colors.BOLD}ε{Colors.END}: {Colors.YELLOW}{epsilon:.4f}{Colors.END}"

    print(info, flush=True)


def print_banner(text: str, width: int = 80) -> None:
    """Print a decorative banner."""
    try:
        from art import text2art

        banner = text2art(text, font="banner")
        print(f"\n{Colors.BOLD}{Colors.GREEN}{banner}{Colors.END}\n", flush=True)
    except ImportError:
        print(
            f"\n{Colors.BOLD}{Colors.GREEN}{'='*width}\n{text.center(width)}\n{'='*width}{Colors.END}\n",
            flush=True,
        )


def print_section_header(title: str, width: int = 80) -> None:
    """Print a section header."""
    separator = "─" * width
    print(f"\n{Colors.BOLD}{Colors.CYAN}{separator}{Colors.END}", flush=True)
    print(f"{Colors.BOLD}{Colors.CYAN}{title.center(width)}{Colors.END}", flush=True)
    print(f"{Colors.BOLD}{Colors.CYAN}{separator}{Colors.END}\n", flush=True)
