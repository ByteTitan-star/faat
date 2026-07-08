#!/media/hd1/wangxin/work7-7month/.conda-envs/GeneralComponents/bin/python3
"""
check_figure.py — Automated figure quality audit for backdoor papers.

Usage:
  python check_figure.py output/figs/asr.pdf              # single figure
  python check_figure.py output/figs/ --all --report r.md  # batch audit
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backdoor-paper-style" / "scripts"))
from backdoor_paper_style import _PROJECT_ROOT

import os
import json
import argparse
from pathlib import Path
from datetime import datetime


def check_single_figure(fig_path):
    """Run automated checks on a single figure file."""
    fig_path = Path(fig_path)
    results = {
        'file': str(fig_path),
        'pass': True,
        'checks': [],
        'warnings': [],
    }

    if not fig_path.exists():
        results['pass'] = False
        results['checks'].append({'check': 'File exists', 'passed': False, 'detail': 'File not found'})
        return results

    # 1. File extension check
    ext = fig_path.suffix.lower()
    checks = []

    # 1a. Has vector version
    has_vector = ext in ('.pdf', '.svg', '.eps')
    checks.append({
        'check': 'Vector format',
        'passed': has_vector,
        'detail': f'Format: {ext} (PDF/SVG/EPS preferred for papers)'
    })
    if not has_vector:
        results['warnings'].append(f'No vector format found. Create PDF version for paper.')

    # 2. File size check
    size_kb = fig_path.stat().st_size / 1024
    size_ok = size_kb < 10000  # < 10 MB
    checks.append({
        'check': 'File size reasonable',
        'passed': size_ok,
        'detail': f'{size_kb:.0f} KB'
    })
    if not size_ok:
        results['warnings'].append(f'File too large ({size_kb:.0f} KB). Consider compressing or reducing dpi.')

    # 3. DPI check (PNG only)
    if ext == '.png':
        try:
            from PIL import Image
            img = Image.open(str(fig_path))
            dpi = img.info.get('dpi', (72, 72))
            dpi_ok = dpi[0] >= 300
            checks.append({
                'check': 'PNG resolution ≥ 300 DPI',
                'passed': dpi_ok,
                'detail': f'{dpi[0]:.0f} DPI'
            })
            if not dpi_ok:
                results['warnings'].append(f'Low DPI ({dpi[0]:.0f}). Use ≥600 DPI for paper figures.')
        except ImportError:
            checks.append({
                'check': 'PNG resolution',
                'passed': None,
                'detail': 'PIL not available; cannot check DPI'
            })

    # 4. Font embedding (PDF only)
    if ext == '.pdf':
        try:
            import subprocess
            pdffonts = subprocess.run(
                ['pdffonts', str(fig_path)],
                capture_output=True, text=True, timeout=10
            )
            if pdffonts.returncode == 0:
                lines = pdffonts.stdout.strip().split('\n')
                # Lines after header; check 'emb' column
                fonts_ok = True
                font_count = 0
                for line in lines[2:]:  # skip header
                    parts = line.split()
                    if len(parts) >= 5:
                        font_count += 1
                        emb = parts[4] if len(parts) > 4 else 'no'
                        if emb == 'no':
                            fonts_ok = False
                checks.append({
                    'check': 'PDF fonts embedded',
                    'passed': fonts_ok,
                    'detail': f'{font_count} fonts, all embedded: {fonts_ok}'
                })
                if not fonts_ok:
                    results['warnings'].append('Some fonts not embedded in PDF.')
            else:
                checks.append({
                    'check': 'PDF font check',
                    'passed': None,
                    'detail': 'pdffonts not available'
                })
        except Exception:
            checks.append({
                'check': 'PDF font check',
                'passed': None,
                'detail': 'Cannot check (pdffonts not available)'
            })

    results['checks'] = checks
    results['pass'] = all(c.get('passed', True) for c in checks if c['passed'] is not None)
    return results


def batch_check(fig_dir, report_path=None):
    """Check all figures in a directory and produce a report."""
    fig_dir = Path(fig_dir)
    exts = {'.pdf', '.png', '.svg', '.jpg', '.jpeg', '.eps'}
    fig_files = sorted([f for f in fig_dir.iterdir() if f.suffix.lower() in exts])

    if not fig_files:
        print(f"No figure files found in {fig_dir}")
        return

    print(f"Checking {len(fig_files)} figures in {fig_dir}...\n")
    all_results = []
    for f in fig_files:
        r = check_single_figure(f)
        all_results.append(r)
        status = '✅' if r['pass'] else '❌'
        print(f"  {status} {f.name}  ({len(r['warnings'])} warnings)")

    # Summary
    passed = sum(1 for r in all_results if r['pass'])
    total_warnings = sum(len(r['warnings']) for r in all_results)
    print(f"\nSummary: {passed}/{len(all_results)} passed, {total_warnings} warnings")

    # Generate report
    if report_path:
        report_path = Path(report_path)
        lines = [
            f"# Figure Quality Report",
            f"",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Directory:** `{fig_dir}`",
            f"**Figures checked:** {len(all_results)}",
            f"**Passed:** {passed}  |  **Warnings:** {total_warnings}",
            f"",
            f"---",
            f"",
            f"## Per-Figure Results",
            f"",
        ]
        for r in all_results:
            status = '✅' if r['pass'] else '❌'
            lines.append(f"### {status} {Path(r['file']).name}")
            for c in r['checks']:
                icon = '✅' if c['passed'] else '❌' if c['passed'] is False else '⚠️'
                lines.append(f"- {icon} **{c['check']}**: {c['detail']}")
            for w in r['warnings']:
                lines.append(f"- ⚠️ {w}")
            lines.append("")

        # Add manual checklist reference
        lines.extend([
            "---",
            "",
            "## Manual Checklist (Review Each Figure)",
            "",
            "See `backdoor-paper-style/checklist.md` for the full checklist. Key items:",
            "",
            "- [ ] Font size ≥ 8pt at final width",
            "- [ ] All axes labeled (quantity + unit)",
            "- [ ] Legend distinguishable in grayscale",
            "- [ ] No chartjunk (3D, gradients, excessive grids)",
            "- [ ] Higher-is-better annotated where ambiguous",
            "- [ ] Error bars present where statistics reported",
            "- [ ] Method ordering consistent across paper",
            "- [ ] Color mapping consistent (use METHOD_COLORS)",
            "- [ ] Proposed method not exaggerated",
            "- [ ] ASR and ACC both shown (backdoor-specific)",
            "- [ ] ACC drop annotated (defense figures)",
            "- [ ] Residual amplification factor annotated (trigger figures)",
            "- [ ] Random seed recorded (t-SNE/UMAP figures)",
            "",
            f"*Report generated by figure-quality-check skill*",
        ])

        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"\nReport saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Audit paper figures for quality issues.")
    parser.add_argument('path', help='Figure file or directory to check')
    parser.add_argument('--all', action='store_true',
                        help='Check all figures in directory')
    parser.add_argument('--report', default=None,
                        help='Save report to markdown file')
    parser.add_argument('--json', default=None,
                        help='Save results as JSON')
    args = parser.parse_args()

    target = Path(args.path)

    if target.is_dir() or args.all:
        batch_check(target, args.report)
    else:
        r = check_single_figure(target)
        status = '✅' if r['pass'] else '❌'
        print(f"{status} {target.name}")
        for c in r['checks']:
            icon = '✅' if c['passed'] else '❌' if c['passed'] is False else '⚠️'
            print(f"  {icon} {c['check']}: {c['detail']}")
        for w in r['warnings']:
            print(f"  ⚠️  {w}")

        if args.json:
            with open(args.json, 'w') as f:
                json.dump(r, f, indent=2)
            print(f"\nJSON saved: {args.json}")


if __name__ == '__main__':
    main()
