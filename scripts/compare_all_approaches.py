#!/usr/bin/env python3
"""
Compare All VQ-VAE Approaches
Unified comparison between baseline VQ-VAE and geodesic/euclidean pipeline results
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_results(experiment_path: str) -> Optional[Dict]:
    """Load results from an experiment directory"""
    exp_dir = Path(experiment_path)
    
    if not exp_dir.exists():
        print(f"Warning: {exp_dir} does not exist")
        return None
    
    # Try different result file formats
    result_files = [
        exp_dir / "evaluation" / "evaluation_results.json",
        exp_dir / "evaluation" / "metrics.yaml",
        exp_dir / "evaluation" / "codebook_health.json",
        exp_dir / "evaluation_results.json",  # Baseline format
        exp_dir / "metrics.yaml",
    ]
    
    results = {}
    
    # Load JSON results
    json_files = [f for f in result_files if f.suffix == '.json' and f.exists()]
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                results.update(data)
                print(f"  Loaded: {json_file}")
        except Exception as e:
            print(f"  Error loading {json_file}: {e}")
    
    # Load YAML results  
    yaml_files = [f for f in result_files if f.suffix == '.yaml' and f.exists()]
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    # Convert string metrics to float
                    for k, v in data.items():
                        if isinstance(v, str):
                            try:
                                data[k] = float(v)
                            except ValueError:
                                pass
                    results.update(data)
                    print(f"  Loaded: {yaml_file}")
        except Exception as e:
            print(f"  Error loading {yaml_file}: {e}")
    
    if not results:
        print(f"  No valid results found in {exp_dir}")
        return None
        
    return results


def extract_metrics(results: Dict, approach_name: str) -> Dict:
    """Extract standardized metrics from results"""
    metrics = {
        "approach": approach_name,
        "model_type": results.get("model_type", approach_name.lower()),
    }
    
    # Extract PSNR (multiple possible locations)
    psnr_candidates = [
        results.get("PSNR"),
        results.get("psnr"),
        results.get("generation_quality", {}).get("psnr"),
        results.get("reconstruction_quality", {}).get("psnr"),
        results.get("psnr_real_vs_quantized"),
    ]
    psnr = next((p for p in psnr_candidates if p is not None), None)
    if psnr is not None:
        metrics["psnr"] = float(psnr)
    
    # Extract SSIM
    ssim_candidates = [
        results.get("SSIM"), 
        results.get("ssim"),
        results.get("generation_quality", {}).get("ssim"),
        results.get("reconstruction_quality", {}).get("ssim"),
        results.get("ssim_real_vs_quantized"),
    ]
    ssim = next((s for s in ssim_candidates if s is not None), None)
    if ssim is not None:
        metrics["ssim"] = float(ssim)
    
    # Extract LPIPS
    lpips_candidates = [
        results.get("LPIPS"),
        results.get("lpips"), 
        results.get("generation_quality", {}).get("lpips"),
    ]
    lpips = next((l for l in lpips_candidates if l is not None), None)
    if lpips is not None:
        metrics["lpips"] = float(lpips)
    
    # Extract codebook metrics
    codebook_info = results.get("codebook_health", results)
    
    entropy_candidates = [
        codebook_info.get("entropy"),
        codebook_info.get("codebook_entropy"),
        results.get("reconstruction_quality", {}).get("codebook_entropy"),
    ]
    entropy = next((e for e in entropy_candidates if e is not None), None)
    if entropy is not None:
        metrics["entropy"] = float(entropy)
    
    usage_candidates = [
        codebook_info.get("usage_percent"),
        codebook_info.get("used_codes", 0) / max(codebook_info.get("codebook_size", 1), 1) * 100,
    ]
    usage = next((u for u in usage_candidates if u is not None), None)
    if usage is not None:
        metrics["usage_percent"] = float(usage)
    
    dead_codes = codebook_info.get("dead_codes", 0)
    if dead_codes is not None:
        metrics["dead_codes"] = int(dead_codes)
    
    codebook_size = codebook_info.get("codebook_size", results.get("codebook_size"))
    if codebook_size is not None:
        metrics["codebook_size"] = int(codebook_size)
    
    return metrics


def create_comparison_table(all_metrics: List[Dict]) -> pd.DataFrame:
    """Create a comparison table from all metrics"""
    df = pd.DataFrame(all_metrics)
    
    # Reorder columns for better display
    column_order = [
        "approach", "model_type", "psnr", "ssim", "lpips", 
        "entropy", "usage_percent", "dead_codes", "codebook_size"
    ]
    
    # Only include columns that exist
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]
    
    return df


def create_visualization(df: pd.DataFrame, out_dir: Path):
    """Create comparison visualizations"""
    if len(df) < 2:
        print("Not enough data for meaningful visualization")
        return
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('VQ-VAE Approaches Comparison - CIFAR-10', fontsize=16, fontweight='bold')
    
    # Color palette for approaches
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8E44AD', '#16A085']
    
    # 1. PSNR Comparison
    if 'psnr' in df.columns and not df['psnr'].isna().all():
        ax = axes[0, 0]
        bars = ax.bar(df['approach'], df['psnr'], color=colors[:len(df)])
        ax.set_title('Peak Signal-to-Noise Ratio (PSNR)', fontweight='bold')
        ax.set_ylabel('PSNR (dB)')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, df['psnr']):
            if not pd.isna(value):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                       f'{value:.2f}', ha='center', va='bottom')
    else:
        axes[0, 0].text(0.5, 0.5, 'PSNR data not available', ha='center', va='center')
        axes[0, 0].set_title('PSNR (Not Available)')
    
    # 2. SSIM Comparison  
    if 'ssim' in df.columns and not df['ssim'].isna().all():
        ax = axes[0, 1]
        bars = ax.bar(df['approach'], df['ssim'], color=colors[:len(df)])
        ax.set_title('Structural Similarity Index (SSIM)', fontweight='bold')
        ax.set_ylabel('SSIM')
        ax.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, df['ssim']):
            if not pd.isna(value):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    else:
        axes[0, 1].text(0.5, 0.5, 'SSIM data not available', ha='center', va='center')
        axes[0, 1].set_title('SSIM (Not Available)')
    
    # 3. LPIPS Comparison (lower is better)
    if 'lpips' in df.columns and not df['lpips'].isna().all():
        ax = axes[1, 0]
        bars = ax.bar(df['approach'], df['lpips'], color=colors[:len(df)])
        ax.set_title('Learned Perceptual Image Patch Similarity (LPIPS)', fontweight='bold')
        ax.set_ylabel('LPIPS (lower is better)')
        ax.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, df['lpips']):
            if not pd.isna(value):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    else:
        axes[1, 0].text(0.5, 0.5, 'LPIPS data not available', ha='center', va='center')
        axes[1, 0].set_title('LPIPS (Not Available)')
    
    # 4. Codebook Usage
    if 'usage_percent' in df.columns and not df['usage_percent'].isna().all():
        ax = axes[1, 1]
        bars = ax.bar(df['approach'], df['usage_percent'], color=colors[:len(df)])
        ax.set_title('Codebook Usage', fontweight='bold')
        ax.set_ylabel('Usage (%)')
        ax.set_ylim(0, 105)
        ax.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, df['usage_percent']):
            if not pd.isna(value):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                       f'{value:.1f}%', ha='center', va='bottom')
    else:
        axes[1, 1].text(0.5, 0.5, 'Usage data not available', ha='center', va='center')
        axes[1, 1].set_title('Codebook Usage (Not Available)')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'comparison_charts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create entropy vs PSNR scatter plot if both available
    if 'entropy' in df.columns and 'psnr' in df.columns and \
       not df['entropy'].isna().all() and not df['psnr'].isna().all():
        
        plt.figure(figsize=(10, 8))
        plt.scatter(df['entropy'], df['psnr'], c=colors[:len(df)], s=100, alpha=0.7)
        
        for i, approach in enumerate(df['approach']):
            plt.annotate(approach, (df['entropy'].iloc[i], df['psnr'].iloc[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Codebook Entropy')
        plt.ylabel('PSNR (dB)')
        plt.title('Codebook Entropy vs Reconstruction Quality', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / 'entropy_vs_psnr.png', dpi=300, bbox_inches='tight')
        plt.close()


def generate_summary_report(df: pd.DataFrame, out_dir: Path):
    """Generate a detailed summary report"""
    report_lines = [
        "# VQ-VAE Approaches Comparison Report - CIFAR-10\n",
        f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"Number of approaches compared: {len(df)}\n\n",
    ]
    
    # Overall summary
    report_lines.append("## Approaches Evaluated\n")
    for _, row in df.iterrows():
        report_lines.append(f"- **{row['approach']}**: {row.get('model_type', 'N/A')}\n")
    report_lines.append("\n")
    
    # Detailed metrics table
    report_lines.append("## Detailed Metrics Comparison\n\n")
    report_lines.append("| Approach | PSNR (dB) | SSIM | LPIPS | Entropy | Usage (%) | Dead Codes |\n")
    report_lines.append("|----------|-----------|------|-------|---------|-----------|------------|\n")
    
    for _, row in df.iterrows():
        psnr = f"{row['psnr']:.2f}" if pd.notna(row.get('psnr')) else "N/A"
        ssim = f"{row['ssim']:.3f}" if pd.notna(row.get('ssim')) else "N/A"
        lpips = f"{row['lpips']:.3f}" if pd.notna(row.get('lpips')) else "N/A"
        entropy = f"{row['entropy']:.2f}" if pd.notna(row.get('entropy')) else "N/A"
        usage = f"{row['usage_percent']:.1f}" if pd.notna(row.get('usage_percent')) else "N/A"
        dead = f"{row['dead_codes']}" if pd.notna(row.get('dead_codes')) else "N/A"
        
        report_lines.append(f"| {row['approach']} | {psnr} | {ssim} | {lpips} | {entropy} | {usage} | {dead} |\n")
    
    report_lines.append("\n")
    
    # Analysis and insights
    report_lines.append("## Analysis\n\n")
    
    # Best performing approaches
    if 'psnr' in df.columns and not df['psnr'].isna().all():
        best_psnr = df.loc[df['psnr'].idxmax()]
        report_lines.append(f"**Best PSNR**: {best_psnr['approach']} ({best_psnr['psnr']:.2f} dB)\n")
    
    if 'ssim' in df.columns and not df['ssim'].isna().all():
        best_ssim = df.loc[df['ssim'].idxmax()]
        report_lines.append(f"**Best SSIM**: {best_ssim['approach']} ({best_ssim['ssim']:.3f})\n")
    
    if 'lpips' in df.columns and not df['lpips'].isna().all():
        best_lpips = df.loc[df['lpips'].idxmin()]  # Lower is better for LPIPS
        report_lines.append(f"**Best LPIPS**: {best_lpips['approach']} ({best_lpips['lpips']:.3f})\n")
    
    if 'entropy' in df.columns and not df['entropy'].isna().all():
        best_entropy = df.loc[df['entropy'].idxmax()]
        report_lines.append(f"**Best Entropy**: {best_entropy['approach']} ({best_entropy['entropy']:.2f})\n")
    
    report_lines.append("\n### Key Insights\n\n")
    report_lines.append("- **Baseline VQ-VAE**: End-to-end trained model with EMA codebook updates\n")
    report_lines.append("- **Euclidean Pipeline**: Post-hoc quantization using standard Euclidean distances\n") 
    report_lines.append("- **Geodesic Pipeline**: Post-hoc quantization using geodesic distances in latent space\n")
    report_lines.append("- **Spatial VAE**: Enhanced architecture with spatial awareness\n\n")
    
    report_lines.append("### Methodology Notes\n\n")
    report_lines.append("- PSNR and SSIM: Higher is better (reconstruction quality)\n")
    report_lines.append("- LPIPS: Lower is better (perceptual similarity)\n")
    report_lines.append("- Entropy: Higher indicates better codebook utilization\n")
    report_lines.append("- Usage %: Percentage of codebook entries actually used\n")
    
    # Save report
    with open(out_dir / "comparison_report.md", 'w') as f:
        f.writelines(report_lines)


def main():
    parser = argparse.ArgumentParser(description="Compare all VQ-VAE approaches")
    parser.add_argument("--out_dir", default="../experiments/cifar10/comparison", 
                       help="Output directory for comparison results")
    parser.add_argument("--approaches", nargs='+', 
                       default=["baseline", "vanilla_euclidean", "vanilla_geodesic", "spatial_geodesic"],
                       help="Approaches to compare")
    
    args = parser.parse_args()
    
    # Setup output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== VQ-VAE APPROACHES COMPARISON ===")
    print(f"Output directory: {out_dir}")
    
    # Define experiment paths (running from scripts/ directory)
    experiment_paths = {
        "baseline": "../experiments/cifar10/baseline_vqvae",
        "vanilla_euclidean": "../experiments/cifar10/vanilla/euclidean", 
        "vanilla_geodesic": "../experiments/cifar10/vanilla/geodesic",
        "spatial_geodesic": "../experiments/cifar10/spatial/geodesic",
    }
    
    # Load results from all approaches
    all_metrics = []
    
    for approach in args.approaches:
        if approach not in experiment_paths:
            print(f"Warning: Unknown approach '{approach}', skipping")
            continue
            
        path = experiment_paths[approach]
        print(f"\nLoading results for: {approach}")
        print(f"  Path: {path}")
        
        results = load_results(path)
        if results is None:
            print(f"  Skipping {approach} - no results found")
            continue
        
        metrics = extract_metrics(results, approach)
        all_metrics.append(metrics)
        print(f"  Extracted metrics: {len(metrics)} fields")
    
    if not all_metrics:
        print("\nNo valid results found for comparison!")
        return 1
    
    print(f"\n=== COMPARISON RESULTS ===")
    print(f"Successfully loaded {len(all_metrics)} approaches")
    
    # Create comparison table
    df = create_comparison_table(all_metrics)
    
    # Display results
    print("\nComparison Table:")
    print("=" * 80)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A'))
    
    # Save detailed results
    df.to_csv(out_dir / "comparison_results.csv", index=False)
    df.to_json(out_dir / "comparison_results.json", orient="records", indent=2)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualization(df, out_dir)
    
    # Generate summary report
    print("Generating summary report...")
    generate_summary_report(df, out_dir)
    
    print(f"\n=== COMPARISON COMPLETE ===")
    print(f"Results saved to: {out_dir}")
    print(f"  Comparison table: comparison_results.csv")
    print(f"  Summary report: comparison_report.md")
    print(f"  Visualizations: comparison_charts.png")
    if (out_dir / "entropy_vs_psnr.png").exists():
        print(f"  Entropy analysis: entropy_vs_psnr.png")
    
    return 0


if __name__ == "__main__":
    exit(main())
