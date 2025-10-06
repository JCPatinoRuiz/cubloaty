"""
cubloaty - Analyze CUDA binary sizes in .so files
Similar to bloaty but for CUDA kernels
"""

import subprocess
import sys
import tempfile
import os
import argparse
import json
from collections import defaultdict
import re

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

def extract_cubins(so_file):
    """Extract cubin sections from .so file"""
    cubins = []

    # Use objcopy to extract .nv_fatbin sections
    try:
        result = subprocess.run(
            ['objdump', '-h', so_file],
            capture_output=True,
            text=True,
            check=True
        )

        # Find all CUDA-related sections
        for line in result.stdout.split('\n'):
            if '.nv_fatbin' in line or 'nv_fatbin' in line:
                # Extract section name
                parts = line.split()
                if len(parts) > 1:
                    section_name = parts[1]
                    cubins.append(section_name)
    except subprocess.CalledProcessError:
        print(f"Error: Could not read sections from {so_file}")
        return []

    return cubins

def extract_cubin_data(so_file, section_name, output_file):
    """Extract cubin binary data from section"""
    try:
        subprocess.run(
            ['objcopy', '--dump-section', f'{section_name}={output_file}', so_file],
            check=True,
            capture_output=True
        )
        return True
    except subprocess.CalledProcessError:
        return False

def extract_cubins_from_fatbin(fatbin_file, output_dir):
    """Extract individual cubins from a fatbin using cuobjdump"""
    try:
        # First, list all ELF files in the fatbin
        result = subprocess.run(
            ['cuobjdump', '-lelf', fatbin_file],
            capture_output=True,
            text=True,
            check=True
        )

        # Parse the output to get cubin names
        cubin_names = []
        for line in result.stdout.split('\n'):
            if 'ELF file' in line:
                # Extract the filename from "ELF file    1: filename.cubin"
                parts = line.split(':', 1)
                if len(parts) == 2:
                    cubin_name = parts[1].strip()
                    cubin_names.append(cubin_name)

        if not cubin_names:
            return []

        # Extract all cubins at once using 'all'
        # cuobjdump will extract them to the current directory
        old_cwd = os.getcwd()
        try:
            os.chdir(output_dir)
            subprocess.run(
                ['cuobjdump', '-xelf', 'all', fatbin_file],
                capture_output=True,
                check=True
            )

            # Find all extracted .cubin files
            extracted_files = []
            for cubin_name in cubin_names:
                cubin_path = os.path.join(output_dir, cubin_name)
                if os.path.exists(cubin_path):
                    extracted_files.append((cubin_name, cubin_path))

            return extracted_files
        finally:
            os.chdir(old_cwd)

    except subprocess.CalledProcessError:
        return []
    except FileNotFoundError:
        print("Error: cuobjdump not found. Please ensure CUDA toolkit is installed and in PATH.")
        return []

def demangle_symbol(symbol):
    """Demangle C++ symbol names"""
    try:
        result = subprocess.run(
            ['c++filt', symbol],
            capture_output=True,
            text=True,
            check=True
        )
        demangled = result.stdout.strip()
        return demangled if demangled else symbol
    except:
        return symbol

def analyze_cubin_sizes(cubin_file):
    """Analyze a single cubin file and return symbol sizes using readelf and size"""
    symbols = {}

    # First, try to get the total text size using the 'size' command
    total_text_size = 0
    try:
        size_result = subprocess.run(
            ['size', cubin_file],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse output: "text\tdata\tbss\tdec\thex\tfilename"
        lines = size_result.stdout.strip().split('\n')
        if len(lines) >= 2:
            parts = lines[1].split()
            if parts:
                total_text_size = int(parts[0])
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass

    # Get function symbols using readelf
    try:
        result = subprocess.run(
            ['readelf', '-sW', cubin_file],
            capture_output=True,
            text=True,
            check=True
        )

        func_total = 0
        # Parse readelf output to extract function names and sizes
        for line in result.stdout.split('\n'):
            # Look for FUNC entries
            if 'FUNC' in line:
                parts = line.split()
                if len(parts) >= 8:
                    try:
                        # The size is typically the 3rd field (index 2)
                        size = int(parts[2], 0)  # 0 base to auto-detect hex/dec
                        # The symbol name is the last part
                        name = parts[-1]
                        if size > 0:  # Only include functions with non-zero size
                            # Demangle the symbol
                            demangled = demangle_symbol(name)
                            symbols[demangled] = size
                            func_total += size
                    except (ValueError, IndexError):
                        continue

        # If we have a text size from 'size' command and it's larger than function symbols,
        # add an entry for unlabeled code
        if total_text_size > func_total and func_total > 0:
            unlabeled_size = total_text_size - func_total
            if unlabeled_size > 0:
                symbols['[other code sections]'] = unlabeled_size

        return symbols
    except subprocess.CalledProcessError:
        return {}
    except FileNotFoundError:
        print("Error: readelf not found. Please ensure binutils is installed.")
        return {}

def format_size(size_bytes):
    """Format size in human-readable format"""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.1f}KB"
    else:
        return f"{size_bytes/(1024*1024):.1f}MB"

def extract_sm_arch(cubin_name):
    """Extract SM architecture from cubin filename (e.g., 'sm_90a' from 'kernel.sm_90a.cubin')"""
    match = re.search(r'\.sm_(\d+[a-z]?)\.cubin', cubin_name)
    if match:
        return f"sm_{match.group(1)}"
    return "unknown"

def get_cubin_arch(cubin_file):
    """Get architecture from cubin file using cuobjdump"""
    try:
        result = subprocess.run(
            ['cuobjdump', '-lelf', cubin_file],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse output like "ELF file    1: kernel.sm_90.cubin"
        for line in result.stdout.split('\n'):
            if 'ELF file' in line and '.sm_' in line:
                match = re.search(r'\.sm_(\d+[a-z]?)\.cubin', line)
                if match:
                    return f"sm_{match.group(1)}"
        return "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to filename-based detection
        return extract_sm_arch(cubin_file)

def shorten_kernel_name(name, max_length=80):
    """Shorten kernel name for display"""
    if len(name) <= max_length:
        return name
    # Try to extract the main function name
    # For templates like ClassName<Args>::method, try to keep the most important part
    if '::' in name:
        parts = name.split('::')
        if len(parts[-1]) < max_length:
            return '...' + '::'.join(parts[-2:])
    return name[:max_length-3] + "..."

def output_json(all_symbols, symbols_by_arch, arch_totals):
    """Output results in JSON format"""
    sorted_symbols = sorted(all_symbols.items(), key=lambda x: x[1], reverse=True)
    total_size = sum(all_symbols.values())

    result = {
        "total_size": total_size,
        "total_size_formatted": format_size(total_size),
        "architectures": {},
        "kernels": []
    }

    # Architecture summary
    for arch in sorted(arch_totals.keys()):
        size = arch_totals[arch]
        percentage = (size / sum(arch_totals.values()) * 100) if sum(arch_totals.values()) > 0 else 0
        result["architectures"][arch] = {
            "size": size,
            "size_formatted": format_size(size),
            "percentage": round(percentage, 2)
        }

    # Overall kernels
    for name, size in sorted_symbols:
        percentage = (size / total_size * 100) if total_size > 0 else 0
        kernel_info = {
            "name": name,
            "size": size,
            "size_formatted": format_size(size),
            "percentage": round(percentage, 2)
        }

        # Add per-arch breakdown if available
        kernel_info["by_arch"] = {}
        for arch in symbols_by_arch:
            if name in symbols_by_arch[arch]:
                kernel_info["by_arch"][arch] = symbols_by_arch[arch][name]

        result["kernels"].append(kernel_info)

    print(json.dumps(result, indent=2))

def main():
    parser = argparse.ArgumentParser(
        description='Analyze CUDA binary sizes in .so files - bloaty for CUDA kernels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cubloaty library.so                    # Analyze CUDA kernels in .so
  cubloaty kernel.cubin                  # Analyze single .cubin file
  cubloaty library.so --top 50           # Show top 50 kernels
  cubloaty library.so --arch sm_90       # Filter by architecture
  cubloaty library.so --filter "gemm"    # Filter kernels by name (regex)
  cubloaty library.so --format json      # Output as JSON
  cubloaty library.so --full-names       # Show full kernel names
        """
    )

    parser.add_argument('file', help='Path to .so or .cubin file to analyze')
    parser.add_argument('--top', '-n', type=int, default=30, metavar='N',
                        help='Show top N kernels (default: 30)')
    parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                        help='Filter by architecture (e.g., sm_90, sm_80)')
    parser.add_argument('--format', '-f', choices=['table', 'json'], default='table',
                        help='Output format (default: table)')
    parser.add_argument('--filter', '-r', type=str, metavar='REGEX',
                        help='Filter kernel names by regular expression (case-insensitive)')
    parser.add_argument('--full-names', action='store_true',
                        help='Show full kernel names without truncation')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored output')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed processing information')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1.0')

    args = parser.parse_args()

    input_file = args.file

    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        sys.exit(1)

    # Determine if we should use rich
    use_rich = not args.no_color and args.format == 'table'
    console = Console() if use_rich else None

    # Check if input is a cubin file
    is_cubin = input_file.endswith('.cubin')

    if args.verbose:
        if console:
            file_type = "cubin file" if is_cubin else "shared library"
            console.print(f"\n[bold cyan]🔍 Analyzing CUDA binaries:[/bold cyan] {os.path.basename(input_file)} ({file_type})")
        else:
            print(f"\nAnalyzing CUDA binaries in: {input_file}")

    # Track symbols by architecture and overall
    symbols_by_arch = defaultdict(lambda: defaultdict(int))
    all_symbols = defaultdict(int)
    arch_totals = defaultdict(int)

    if is_cubin:
        # Direct cubin analysis
        if args.verbose:
            if console:
                console.print("[yellow]📦 Processing cubin file...[/yellow]")
            else:
                print("Processing cubin file...")

        # Try to analyze directly
        symbols = analyze_cubin_sizes(input_file)

        if symbols:
            if args.verbose:
                if console:
                    console.print(f"[green]✓[/green] Found {len(symbols)} kernel(s)")
                else:
                    print(f"Found {len(symbols)} kernel(s)")

            # Extract architecture from cubin file
            arch = get_cubin_arch(input_file)

            for name, size in symbols.items():
                all_symbols[name] += size
                symbols_by_arch[arch][name] += size
                arch_totals[arch] += size
        else:
            print("No symbols found in cubin file")
            sys.exit(1)
    else:
        # .so file processing (existing logic)
        # Extract cubin sections
        sections = extract_cubins(input_file)

        if not sections:
            print("No CUDA binary sections found in the file.")
            if args.verbose:
                print("Trying to extract using cuobjdump...")
                try:
                    subprocess.run(['cuobjdump', '-elf', input_file], check=True)
                    print("Use cuobjdump -elf <file> to extract cubins manually")
                except:
                    pass
            sys.exit(1)

        if args.verbose:
            if console:
                console.print(f"[green]✓[/green] Found {len(sections)} CUDA binary section(s)")
            else:
                print(f"Found {len(sections)} CUDA binary section(s)\n")

        # Process each section
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, section in enumerate(sections):
                if args.verbose:
                    if console:
                        console.print(f"\n[bold]📦 Processing section:[/bold] {section}")
                    else:
                        print(f"\nProcessing section: {section}")

                cubin_file = os.path.join(tmpdir, f'cubin_{i}.bin')

                if not extract_cubin_data(input_file, section, cubin_file):
                    if args.verbose:
                        print(f"  Warning: Could not extract {section}")
                    continue

                # First try to disassemble directly
                symbols = analyze_cubin_sizes(cubin_file)

                # If that failed, it might be a fatbin, try extracting cubins from it
                if not symbols:
                    if args.verbose:
                        if console:
                            console.print("  [yellow]🔄 Extracting cubins from fatbin...[/yellow]")
                        else:
                            print("  Attempting to extract cubins from fatbin...")
                    extracted_cubins = extract_cubins_from_fatbin(cubin_file, tmpdir)

                    if extracted_cubins:
                        if args.verbose:
                            if console:
                                console.print(f"  [green]✓[/green] Found {len(extracted_cubins)} cubin(s)")
                            else:
                                print(f"  Found {len(extracted_cubins)} cubin(s) in fatbin")

                        # Group by architecture
                        arch_groups = defaultdict(list)
                        for cubin_name, cubin_path in extracted_cubins:
                            arch = extract_sm_arch(cubin_name)
                            arch_groups[arch].append((cubin_name, cubin_path))

                        for arch in sorted(arch_groups.keys()):
                            cubins = arch_groups[arch]
                            arch_kernel_count = 0
                            for cubin_name, cubin_path in cubins:
                                cubin_symbols = analyze_cubin_sizes(cubin_path)
                                if cubin_symbols:
                                    arch_kernel_count += len(cubin_symbols)
                                    for name, size in cubin_symbols.items():
                                        symbols_by_arch[arch][name] += size
                                        all_symbols[name] += size
                                        arch_totals[arch] += size

                            if args.verbose:
                                if console:
                                    console.print(f"    [cyan]{arch}[/cyan]: {len(cubins)} cubin(s), {arch_kernel_count} kernel(s), {format_size(arch_totals[arch])}")
                                else:
                                    print(f"    {arch}: {len(cubins)} cubin(s), {format_size(arch_totals[arch])}")
                    else:
                        if args.verbose:
                            print(f"  No symbols found in {section}")
                    continue

                if args.verbose:
                    if console:
                        console.print(f"  [green]✓[/green] Found {len(symbols)} kernel(s)")
                    else:
                        print(f"  Found {len(symbols)} kernel(s)")

                for name, size in symbols.items():
                    all_symbols[name] += size

    # Filter by architecture if specified
    if args.arch:
        if args.arch not in symbols_by_arch:
            print(f"Error: Architecture '{args.arch}' not found. Available: {', '.join(sorted(symbols_by_arch.keys()))}")
            sys.exit(1)
        # Replace all_symbols with filtered symbols
        all_symbols = symbols_by_arch[args.arch]
        # Keep only the requested arch
        symbols_by_arch = {args.arch: symbols_by_arch[args.arch]}
        arch_totals = {args.arch: arch_totals[args.arch]}

    # Filter by regex pattern if specified
    if args.filter:
        try:
            pattern = re.compile(args.filter, re.IGNORECASE)
        except re.error as e:
            print(f"Error: Invalid regular expression: {e}")
            sys.exit(1)

        # Count before filtering
        total_before = len(all_symbols)

        # Filter all_symbols
        all_symbols = {name: size for name, size in all_symbols.items() if pattern.search(name)}

        # Filter symbols_by_arch
        for arch in symbols_by_arch:
            symbols_by_arch[arch] = {name: size for name, size in symbols_by_arch[arch].items() if pattern.search(name)}
            # Recalculate arch totals
            arch_totals[arch] = sum(symbols_by_arch[arch].values())

        if args.verbose:
            matched = len(all_symbols)
            if console:
                console.print(f"\n[yellow]📋 Filter:[/yellow] Matched {matched}/{total_before} kernels with pattern '{args.filter}'")
            else:
                print(f"\nFilter: Matched {matched}/{total_before} kernels with pattern '{args.filter}'")

        if not all_symbols:
            print(f"No kernels matched the filter pattern '{args.filter}'")
            sys.exit(0)

    # Output based on format
    if args.format == 'json':
        output_json(all_symbols, symbols_by_arch, arch_totals)
        return

    # Print results using rich tables
    if console and use_rich:
        console.print()
        console.print(Panel.fit("[bold cyan]📊 CUDA Kernel Size Analysis Report[/bold cyan]", border_style="cyan"))

        # Architecture summary table
        if arch_totals:
            arch_table = Table(title="Architecture Summary", box=box.ROUNDED, show_header=True, header_style="bold magenta")
            arch_table.add_column("Architecture", style="cyan", width=15)
            arch_table.add_column("Total Size", justify="right", style="yellow", width=15)
            arch_table.add_column("Percentage", justify="right", style="green", width=12)

            total_all_arch = sum(arch_totals.values())
            for arch in sorted(arch_totals.keys()):
                size = arch_totals[arch]
                percentage = (size / total_all_arch * 100) if total_all_arch > 0 else 0
                arch_table.add_row(arch.upper(), format_size(size), f"{percentage:.1f}%")

            arch_table.add_section()
            arch_table.add_row("[bold]TOTAL[/bold]", f"[bold]{format_size(total_all_arch)}[/bold]", "[bold]100.0%[/bold]")
            console.print(arch_table)
            console.print()

        # Overall top kernels table
        title = "Top Kernels (All Architectures)" if not args.arch else f"Top Kernels ({args.arch.upper()})"
        if args.filter:
            title += f" - Filter: '{args.filter}'"
        kernel_table = Table(title=title, box=box.ROUNDED, show_header=True, header_style="bold magenta")
        kernel_table.add_column("Rank", style="dim", width=6, justify="right")
        name_width = 120 if args.full_names else 70
        kernel_table.add_column("Kernel Name", style="cyan", width=name_width)
        kernel_table.add_column("Total Size", justify="right", style="yellow", width=12)
        kernel_table.add_column("%", justify="right", style="green", width=8)

        sorted_symbols = sorted(all_symbols.items(), key=lambda x: x[1], reverse=True)
        total_size = sum(all_symbols.values())

        # Show top N kernels
        display_count = min(args.top, len(sorted_symbols))
        for idx, (name, size) in enumerate(sorted_symbols[:display_count], 1):
            percentage = (size / total_size * 100) if total_size > 0 else 0
            short_name = name if args.full_names else shorten_kernel_name(name, name_width)
            kernel_table.add_row(str(idx), short_name, format_size(size), f"{percentage:.1f}%")

        if len(sorted_symbols) > display_count:
            kernel_table.add_row("...", f"[dim]({len(sorted_symbols) - display_count} more kernels)[/dim]", "", "")

        kernel_table.add_section()
        kernel_table.add_row("", "[bold]TOTAL[/bold]", f"[bold]{format_size(total_size)}[/bold]", "[bold]100.0%[/bold]")
        console.print(kernel_table)

        # Per-architecture breakdown (only if not filtering and multiple archs)
        if not args.arch and len(symbols_by_arch) > 1:
            for arch in sorted(symbols_by_arch.keys()):
                console.print()
                arch_symbols = symbols_by_arch[arch]
                arch_sorted = sorted(arch_symbols.items(), key=lambda x: x[1], reverse=True)
                arch_total = sum(arch_symbols.values())

                per_arch_table = Table(title=f"Kernels for {arch.upper()}", box=box.ROUNDED, show_header=True, header_style="bold magenta")
                per_arch_table.add_column("Rank", style="dim", width=6, justify="right")
                per_arch_table.add_column("Kernel Name", style="cyan", width=name_width)
                per_arch_table.add_column("Size", justify="right", style="yellow", width=12)
                per_arch_table.add_column("%", justify="right", style="green", width=8)

                # Show top 15 per architecture
                arch_display = min(15, len(arch_sorted))
                for idx, (name, size) in enumerate(arch_sorted[:arch_display], 1):
                    percentage = (size / arch_total * 100) if arch_total > 0 else 0
                    short_name = name if args.full_names else shorten_kernel_name(name, name_width)
                    per_arch_table.add_row(str(idx), short_name, format_size(size), f"{percentage:.1f}%")

                if len(arch_sorted) > arch_display:
                    per_arch_table.add_row("...", f"[dim]({len(arch_sorted) - arch_display} more kernels)[/dim]", "", "")

                per_arch_table.add_section()
                per_arch_table.add_row("", "[bold]TOTAL[/bold]", f"[bold]{format_size(arch_total)}[/bold]", "[bold]100.0%[/bold]")
                console.print(per_arch_table)

        console.print("\n[bold green]✓ Analysis complete![/bold green]\n")
    else:
        # Fallback to basic output
        print("\n" + "="*100)
        print("CUDA Kernel Size Report")
        print("="*100)

        sorted_symbols = sorted(all_symbols.items(), key=lambda x: x[1], reverse=True)
        total_size = sum(all_symbols.values())

        name_width = 90 if args.full_names else 70
        print(f"\n{'Kernel Name':<{name_width}} {'Size':>15} {'%':>10}")
        print("-"*100)

        display_count = min(args.top, len(sorted_symbols))
        for name, size in sorted_symbols[:display_count]:
            percentage = (size / total_size * 100) if total_size > 0 else 0
            short_name = name if args.full_names else shorten_kernel_name(name, name_width)
            print(f"{short_name:<{name_width}} {format_size(size):>15} {percentage:>9.1f}%")

        if len(sorted_symbols) > display_count:
            print(f"... ({len(sorted_symbols) - display_count} more kernels)")

        print("-"*100)
        print(f"{'TOTAL':<{name_width}} {format_size(total_size):>15} {'100.0%':>10}")
        print()

if __name__ == '__main__':
    main()
