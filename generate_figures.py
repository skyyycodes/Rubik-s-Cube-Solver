#!/usr/bin/env python3
"""Generate publication-quality figures for the Rubik's Cube solver paper.

Produces:
  figures/fig_move_distribution.pdf   – Move-count distribution K=1 vs K=5
  figures/fig_node_expansion.pdf      – Node expansion bar chart by strategy
  figures/fig_anytime_curve.pdf       – Quality vs time budget (anytime)
  figures/fig_neural_comparison.pdf   – Neural strategy comparison (mean moves)

Usage:
    python3 generate_figures.py
"""

import json
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Style ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

COLORS = {
    'baseline': '#d62728',   # red
    'k5':       '#1f77b4',   # blue
    'neural_a': '#ff7f0e',   # orange
    'neural_b': '#2ca02c',   # green
    'neural_c': '#9467bd',   # purple
    'accent':   '#17becf',   # teal
}

os.makedirs('figures', exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: Move-count distribution (K=1 vs K=5)
# ═══════════════════════════════════════════════════════════════════════
def fig_move_distribution():
    data = load_json('results/benchmark.json')
    bl = data['baseline']
    imp = data['improved']

    # Reconstruct distribution from summary stats
    # We have: total=200, distribution: optimal_20 (<=20), near_optimal_18 (<=18), good_21 (<=21)
    # Also have min, max, mean, median, stdev
    # For a histogram we need per-move counts. Let's compute from the distribution buckets.
    # optimal_20 = count ≤ 20, near_optimal_18 = count ≤ 18, good_21 = count ≤ 21
    n = bl['total_scrambles']

    def dist_to_counts(d, n):
        """Convert cumulative distribution fields to per-move-count bins."""
        le18 = d['near_optimal_18']
        le20 = d['optimal_20']
        le21 = d['good_21']
        # counts: 18, 19, 20, 21, 22
        c18 = le18
        c19 = le20 - le18   # This is count of 19 + 20; need to split further
        # Actually optimal_20 means count with ≤20 moves
        # near_optimal_18 means count with ≤18 moves
        # good_21 means count with ≤21 moves
        # So: c_le18 = near_optimal_18, c_le20 = optimal_20, c_le21 = good_21
        # c19_20 = c_le20 - c_le18 (moves 19 or 20)
        # c21 = c_le21 - c_le20
        # c22 = n - c_le21
        c_le18 = le18
        c_19_20 = le20 - le18
        c_21 = le21 - le20
        c_22 = n - le21
        return c_le18, c_19_20, c_21, c_22

    bl_counts = dist_to_counts(bl['distribution'], n)
    imp_counts = dist_to_counts(imp['distribution'], n)

    labels = ['≤18', '19-20', '21', '22']
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5, 3.2))
    bars1 = ax.bar(x - width/2, bl_counts, width, label='Baseline (K=1)',
                   color=COLORS['baseline'], alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + width/2, imp_counts, width, label='Multi-solution (K=5)',
                   color=COLORS['k5'], alpha=0.85, edgecolor='white')

    ax.set_xlabel('Move Count Range')
    ax.set_ylabel('Number of Solves (out of 200)')
    ax.set_title('Solution Length Distribution: K=1 vs K=5')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.set_ylim(0, max(max(bl_counts), max(imp_counts)) * 1.15)

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, str(int(h)),
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, str(int(h)),
                    ha='center', va='bottom', fontsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig('figures/fig_move_distribution.pdf')
    fig.savefig('figures/fig_move_distribution.png')
    plt.close(fig)
    print('  ✓ fig_move_distribution.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Node expansion by strategy
# ═══════════════════════════════════════════════════════════════════════
def fig_node_expansion():
    data = load_json('results/neural_strategies.json')
    sc = data['strategy_comparison']

    strategies = ['baseline_k1', 'K5-none', 'K5-move_order', 'K5-probabilistic', 'K5-adaptive']
    labels = ['Baseline\n(K=1)', 'K=5\nNo Neural', 'K=5\nStrategy A', 'K=5\nStrategy B', 'K=5\nStrategy C']
    colors_list = [COLORS['baseline'], COLORS['k5'], COLORS['neural_a'], COLORS['neural_b'], COLORS['neural_c']]

    phase1 = [sc[s]['node_stats']['mean_phase1_nodes'] / 1e6 for s in strategies]
    phase2 = [sc[s]['node_stats']['mean_phase2_nodes'] / 1e6 for s in strategies]

    x = np.arange(len(strategies))
    width = 0.6

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars1 = ax.bar(x, phase1, width, label='Phase 1', color=colors_list, alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x, phase2, width, bottom=phase1, label='Phase 2',
                   color=[c + '80' if len(c) == 7 else c for c in colors_list],
                   alpha=0.5, edgecolor='white', hatch='///')

    ax.set_ylabel('Mean Nodes Expanded (millions)')
    ax.set_title('Node Expansion by Strategy')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(frameon=True, fancybox=False, edgecolor='gray', loc='upper left')

    # Total labels
    for i, (p1, p2) in enumerate(zip(phase1, phase2)):
        total = p1 + p2
        ax.text(i, total + 0.05, f'{total:.1f}M', ha='center', va='bottom', fontsize=7.5,
                fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max(p1+p2 for p1, p2 in zip(phase1, phase2)) * 1.15)
    fig.savefig('figures/fig_node_expansion.pdf')
    fig.savefig('figures/fig_node_expansion.png')
    plt.close(fig)
    print('  ✓ fig_node_expansion.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: Anytime quality vs time budget
# ═══════════════════════════════════════════════════════════════════════
def fig_anytime_curve():
    data = load_json('results/anytime_sweep.json')
    sweep = data['time_budget_sweep']

    budgets = sorted(sweep.keys(), key=float)
    budget_vals = [float(b) for b in budgets]
    mean_moves = [sweep[b]['mean_moves'] for b in budgets]
    success_rate = [sweep[b]['successful'] / sweep[b]['total'] * 100 for b in budgets]
    pct_le20 = [sweep[b]['pct_le20'] for b in budgets]

    fig, ax1 = plt.subplots(figsize=(5, 3.2))
    ax2 = ax1.twinx()

    # Success rate (left axis)
    line1, = ax1.plot(budget_vals, success_rate, 'o-', color=COLORS['k5'],
                      linewidth=2, markersize=6, label='Success rate (%)')
    ax1.set_xlabel('Time Budget (seconds)')
    ax1.set_ylabel('Success Rate (%)', color=COLORS['k5'])
    ax1.tick_params(axis='y', labelcolor=COLORS['k5'])
    ax1.set_ylim(0, 105)

    # Mean moves (right axis)
    line2, = ax2.plot(budget_vals, mean_moves, 's--', color=COLORS['baseline'],
                      linewidth=2, markersize=6, label='Mean moves')
    ax2.set_ylabel('Mean Move Count', color=COLORS['baseline'])
    ax2.tick_params(axis='y', labelcolor=COLORS['baseline'])
    ax2.set_ylim(19.8, 20.6)

    ax1.set_title('Anytime Search: Quality vs Time Budget (K=5)')
    lines = [line1, line2]
    labels_legend = [l.get_label() for l in lines]
    ax1.legend(lines, labels_legend, loc='center right', frameon=True,
               fancybox=False, edgecolor='gray')

    # Annotate success counts
    for bv, sr, succ_b in zip(budget_vals, success_rate, budgets):
        succ = sweep[succ_b]['successful']
        tot = sweep[succ_b]['total']
        ax1.annotate(f'{succ}/{tot}', (bv, sr), textcoords='offset points',
                     xytext=(0, 10), ha='center', fontsize=7.5, color=COLORS['k5'])

    ax1.spines['top'].set_visible(False)
    fig.savefig('figures/fig_anytime_curve.pdf')
    fig.savefig('figures/fig_anytime_curve.png')
    plt.close(fig)
    print('  ✓ fig_anytime_curve.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: Neural strategy comparison (mean moves + ≤20%)
# ═══════════════════════════════════════════════════════════════════════
def fig_neural_comparison():
    # Original model
    d1 = load_json('results/neural_strategies.json')
    sc1 = d1['strategy_comparison']
    # Retrained model
    d2 = load_json('results/neural_strategies_retrained.json')
    sc2 = d2['strategy_comparison']

    strategies_k5 = ['K5-none', 'K5-move_order', 'K5-probabilistic', 'K5-adaptive']
    labels = ['No Neural', 'A: Move Order', 'B: Probabilistic', 'C: Adaptive']

    # Original model data (200 scrambles)
    orig_mean = [sc1[s]['move_stats']['mean'] for s in strategies_k5]
    orig_le20 = [sc1[s]['distribution']['optimal_20'] for s in strategies_k5]
    orig_n = sc1['K5-none']['total_scrambles']

    # Retrained model data (50 scrambles)
    retr_mean = [sc2[s]['move_stats']['mean'] for s in strategies_k5]
    retr_le20_raw = [sc2[s]['distribution']['optimal_20'] for s in strategies_k5]
    retr_n = sc2['K5-none']['total_scrambles']
    # Normalize to percentage for fair comparison
    orig_le20_pct = [x / orig_n * 100 for x in orig_le20]
    retr_le20_pct = [x / retr_n * 100 for x in retr_le20_raw]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))

    x = np.arange(len(labels))
    width = 0.35

    # Panel A: Mean moves
    ax1.bar(x - width/2, orig_mean, width, label='Original Model\n(~5.6% acc)',
            color=COLORS['k5'], alpha=0.85, edgecolor='white')
    ax1.bar(x + width/2, retr_mean, width, label='Retrained Model\n(3.9% acc)',
            color=COLORS['neural_a'], alpha=0.85, edgecolor='white')
    ax1.axhline(y=sc1['baseline_k1']['move_stats']['mean'], color=COLORS['baseline'],
                linestyle='--', linewidth=1, label=f'Baseline K=1 ({sc1["baseline_k1"]["move_stats"]["mean"]:.2f})')
    ax1.set_ylabel('Mean Move Count')
    ax1.set_title('(a) Mean Moves by Strategy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7.5, rotation=15, ha='right')
    ax1.set_ylim(20.0, 21.0)
    ax1.legend(fontsize=7, frameon=True, fancybox=False, edgecolor='gray')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: ≤20 percentage
    ax2.bar(x - width/2, orig_le20_pct, width, label='Original Model',
            color=COLORS['k5'], alpha=0.85, edgecolor='white')
    ax2.bar(x + width/2, retr_le20_pct, width, label='Retrained Model',
            color=COLORS['neural_a'], alpha=0.85, edgecolor='white')
    bl_le20_pct = sc1['baseline_k1']['distribution']['optimal_20'] / orig_n * 100
    ax2.axhline(y=bl_le20_pct, color=COLORS['baseline'],
                linestyle='--', linewidth=1, label=f'Baseline K=1 ({bl_le20_pct:.0f}%)')
    ax2.set_ylabel('Solutions ≤20 Moves (%)')
    ax2.set_title('(b) Optimal Solutions by Strategy')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7.5, rotation=15, ha='right')
    ax2.set_ylim(0, 80)
    ax2.legend(fontsize=7, frameon=True, fancybox=False, edgecolor='gray')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('Neural Strategies: Both Training Paradigms Fail', fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig('figures/fig_neural_comparison.pdf')
    fig.savefig('figures/fig_neural_comparison.png')
    plt.close(fig)
    print('  ✓ fig_neural_comparison.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Figure 5: Timing distribution (K=1 vs K=5) — log-scale histogram
# ═══════════════════════════════════════════════════════════════════════
def fig_timing_distribution():
    """Visualise the right-skewed solve-time distribution for K=1 vs K=5.

    Since we only have summary statistics (mean, median, max) we draw a
    schematic comparison showing key percentiles as annotated markers on
    a horizontal bar/lollipop chart.  This is more honest than faking a
    histogram from three data points.
    """
    data = load_json('results/benchmark.json')
    bl_t = data['baseline']['time_stats']
    imp_t = data['improved']['time_stats']
    # Also grab K=3 and K=10 for a richer picture
    k3_t = load_json('results/benchmark_3.json')['improved']['time_stats']
    k10_t = load_json('results/benchmark_10.json')['improved']['time_stats']

    configs = [
        ('K=1\n(Baseline)', bl_t, COLORS['baseline']),
        ('K=3', k3_t, '#ff7f0e'),
        ('K=5', imp_t, COLORS['k5']),
        ('K=10', k10_t, COLORS['neural_c']),
    ]

    fig, ax = plt.subplots(figsize=(5, 3.0))
    y_positions = np.arange(len(configs))

    for i, (label, ts, color) in enumerate(configs):
        median = ts['median_ms']
        mean = ts['mean_ms']
        mx = ts['max_ms']

        # Draw a line from 0 to max
        ax.plot([0, mx], [i, i], color=color, linewidth=1.5, alpha=0.4)
        # Median marker
        ax.plot(median, i, 'o', color=color, markersize=8, zorder=5)
        # Mean marker
        ax.plot(mean, i, 'D', color=color, markersize=6, zorder=5,
                markeredgecolor='white', markeredgewidth=0.5)
        # Max marker
        ax.plot(mx, i, '|', color=color, markersize=12, markeredgewidth=2, zorder=5)

        # Annotate values
        ax.annotate(f'med={median:.0f}', (median, i), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=6.5, color=color)
        ax.annotate(f'mean={mean:.0f}', (mean, i), textcoords='offset points',
                    xytext=(0, -14), ha='center', fontsize=6.5, color=color)
        if mx > mean * 1.5:
            ax.annotate(f'max={mx:.0f}', (mx, i), textcoords='offset points',
                        xytext=(0, 10), ha='right', fontsize=6.5, color=color)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([c[0] for c in configs])
    ax.set_xlabel('Solve Time (ms)')
    ax.set_title('Solve Time Distribution by K-value')
    ax.set_xscale('log')
    ax.set_xlim(50, 100_000)

    # Custom legend for marker types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='gray', linestyle='None',
               markersize=7, label='Median'),
        Line2D([0], [0], marker='D', color='gray', linestyle='None',
               markersize=5, label='Mean'),
        Line2D([0], [0], marker='|', color='gray', linestyle='None',
               markersize=10, markeredgewidth=2, label='Max'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True,
              fancybox=False, edgecolor='gray', fontsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig('figures/fig_timing_distribution.pdf')
    fig.savefig('figures/fig_timing_distribution.png')
    plt.close(fig)
    print('  ✓ fig_timing_distribution.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Figure 6: K-value scaling (mean moves, ≤20%, solve time)
# ═══════════════════════════════════════════════════════════════════════
def fig_k_scaling():
    """Three-panel figure showing how K affects quality and cost."""
    bm = load_json('results/benchmark.json')
    bm3 = load_json('results/benchmark_3.json')
    bm10 = load_json('results/benchmark_10.json')

    configs = [
        (1, bm['baseline']),
        (3, bm3['improved']),
        (5, bm['improved']),
        (10, bm10['improved']),
    ]

    ks = [c[0] for c in configs]
    mean_moves = [c[1]['move_stats']['mean'] for c in configs]
    le20_pct = [c[1]['distribution']['optimal_20'] / c[1]['total_scrambles'] * 100
                for c in configs]
    mean_time = [c[1]['time_stats']['mean_ms'] for c in configs]
    success = [c[1]['successful'] / c[1]['total_scrambles'] * 100 for c in configs]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 2.8))

    # Panel A: Mean moves
    ax1.plot(ks, mean_moves, 'o-', color=COLORS['k5'], linewidth=2, markersize=7)
    ax1.set_xlabel('K (Phase 1 solutions)')
    ax1.set_ylabel('Mean Move Count')
    ax1.set_title('(a) Solution Quality')
    ax1.set_xticks(ks)
    ax1.set_ylim(20.0, 21.0)
    ax1.axhline(y=20, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    for k, m in zip(ks, mean_moves):
        ax1.annotate(f'{m:.2f}', (k, m), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=7.5, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: ≤20 percentage
    ax2.plot(ks, le20_pct, 's-', color=COLORS['neural_a'], linewidth=2, markersize=7)
    ax2.set_xlabel('K (Phase 1 solutions)')
    ax2.set_ylabel('Solutions ≤20 Moves (%)')
    ax2.set_title('(b) Optimal Rate')
    ax2.set_xticks(ks)
    ax2.set_ylim(20, 70)
    for k, p in zip(ks, le20_pct):
        ax2.annotate(f'{p:.1f}%', (k, p), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=7.5, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Panel C: Solve time (log scale)
    ax3.plot(ks, mean_time, '^-', color=COLORS['baseline'], linewidth=2, markersize=7)
    ax3.set_xlabel('K (Phase 1 solutions)')
    ax3.set_ylabel('Mean Solve Time (ms)')
    ax3.set_title('(c) Computational Cost')
    ax3.set_xticks(ks)
    ax3.set_yscale('log')
    for k, t in zip(ks, mean_time):
        ax3.annotate(f'{t:.0f}', (k, t), textcoords='offset points',
                     xytext=(0, 10), ha='center', fontsize=7.5, fontweight='bold')
    # Mark 100% success rate floor
    for k, s_rate in zip(ks, success):
        if s_rate < 100:
            ax3.annotate(f'{s_rate:.0f}%\nsuccess', (k, mean_time[ks.index(k)]),
                         textcoords='offset points', xytext=(0, -22),
                         ha='center', fontsize=6.5, color=COLORS['baseline'])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    fig.suptitle('Effect of K on Solution Quality and Cost', fontsize=11,
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig('figures/fig_k_scaling.pdf')
    fig.savefig('figures/fig_k_scaling.png')
    plt.close(fig)
    print('  ✓ fig_k_scaling.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Figure 7: Multi-seed consistency
# ═══════════════════════════════════════════════════════════════════════
def fig_multi_seed():
    """Show improvement consistency across 4 random seeds."""
    seeds_data = {}
    for s in [42, 123, 456, 789]:
        fn = 'results/benchmark.json' if s == 42 else f'results/benchmark_seed{s}.json'
        seeds_data[s] = load_json(fn)

    seeds = [42, 123, 456, 789]
    bl_means = [seeds_data[s]['baseline']['move_stats']['mean'] for s in seeds]
    k5_means = [seeds_data[s]['improved']['move_stats']['mean'] for s in seeds]
    deltas = [bl - k5 for bl, k5 in zip(bl_means, k5_means)]
    bl_le20 = [seeds_data[s]['baseline']['distribution']['optimal_20'] /
               seeds_data[s]['baseline']['total_scrambles'] * 100 for s in seeds]
    k5_le20 = [seeds_data[s]['improved']['distribution']['optimal_20'] /
               seeds_data[s]['improved']['total_scrambles'] * 100 for s in seeds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.0))
    x = np.arange(len(seeds))
    width = 0.35
    seed_labels = [f'Seed {s}' for s in seeds]

    # Panel A: Mean moves
    bars1 = ax1.bar(x - width/2, bl_means, width, label='Baseline (K=1)',
                    color=COLORS['baseline'], alpha=0.85, edgecolor='white')
    bars2 = ax1.bar(x + width/2, k5_means, width, label='K=5',
                    color=COLORS['k5'], alpha=0.85, edgecolor='white')
    ax1.set_xlabel('Random Seed')
    ax1.set_ylabel('Mean Move Count')
    ax1.set_title('(a) Mean Moves by Seed')
    ax1.set_xticks(x)
    ax1.set_xticklabels(seed_labels, fontsize=8)
    ax1.set_ylim(19.8, 21.2)
    ax1.legend(fontsize=7.5, frameon=True, fancybox=False, edgecolor='gray')
    # Add delta annotations
    for i, d in enumerate(deltas):
        y_mid = (bl_means[i] + k5_means[i]) / 2
        ax1.annotate(f'Δ={d:.2f}', (i, y_mid), textcoords='offset points',
                     xytext=(18, 0), ha='left', fontsize=7, color='#333333',
                     arrowprops=dict(arrowstyle='-', color='#999999', lw=0.5))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: ≤20 percentage
    bars3 = ax2.bar(x - width/2, bl_le20, width, label='Baseline (K=1)',
                    color=COLORS['baseline'], alpha=0.85, edgecolor='white')
    bars4 = ax2.bar(x + width/2, k5_le20, width, label='K=5',
                    color=COLORS['k5'], alpha=0.85, edgecolor='white')
    ax2.set_xlabel('Random Seed')
    ax2.set_ylabel('Solutions ≤20 Moves (%)')
    ax2.set_title('(b) Optimal Rate by Seed')
    ax2.set_xticks(x)
    ax2.set_xticklabels(seed_labels, fontsize=8)
    ax2.set_ylim(0, 75)
    ax2.legend(fontsize=7.5, frameon=True, fancybox=False, edgecolor='gray')
    # Add value labels
    for bar_set in [bars3, bars4]:
        for bar in bar_set:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.0f}%',
                     ha='center', va='bottom', fontsize=7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('Multi-Seed Validation: Consistent Improvement Across Seeds',
                 fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig('figures/fig_multi_seed.pdf')
    fig.savefig('figures/fig_multi_seed.png')
    plt.close(fig)
    print('  ✓ fig_multi_seed.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Figure 8: Hard-case comparison
# ═══════════════════════════════════════════════════════════════════════
def fig_hard_cases():
    """Compare baseline vs K=5 on hard scrambles — move distribution and time."""
    hc = load_json('results/hard_cases.json')
    bl = hc['baseline']
    imp = hc['improved']
    n = bl['successful']  # 49 solved

    def dist_to_counts(d, n_succ):
        le18 = d['near_optimal_18']
        le20 = d['optimal_20']
        le21 = d['good_21']
        c_le18 = le18
        c_19_20 = le20 - le18
        c_21 = le21 - le20
        c_22 = n_succ - le21
        return c_le18, c_19_20, c_21, c_22

    bl_counts = dist_to_counts(bl['distribution'], bl['successful'])
    imp_counts = dist_to_counts(imp['distribution'], imp['successful'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.0))

    # Panel A: Move distribution
    labels = ['≤18', '19-20', '21', '22+']
    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax1.bar(x - width/2, bl_counts, width, label='Baseline (K=1)',
                    color=COLORS['baseline'], alpha=0.85, edgecolor='white')
    bars2 = ax1.bar(x + width/2, imp_counts, width, label='Multi-solution (K=5)',
                    color=COLORS['k5'], alpha=0.85, edgecolor='white')
    ax1.set_xlabel('Move Count Range')
    ax1.set_ylabel('Number of Solves (out of 49)')
    ax1.set_title('(a) Hard-Case Move Distribution')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(fontsize=7.5, frameon=True, fancybox=False, edgecolor='gray')
    ax1.set_ylim(0, max(max(bl_counts), max(imp_counts)) * 1.25)
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            h = bar.get_height()
            if h > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, h + 0.3, str(int(h)),
                         ha='center', va='bottom', fontsize=7.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: Key metrics comparison
    metrics = ['Mean\nMoves', '≤20\nMoves (%)', 'Mean Time\n(seconds)']
    bl_vals = [
        bl['move_stats']['mean'],
        bl['distribution']['optimal_20'] / bl['successful'] * 100,
        bl['time_stats']['mean_ms'] / 1000,
    ]
    imp_vals = [
        imp['move_stats']['mean'],
        imp['distribution']['optimal_20'] / imp['successful'] * 100,
        imp['time_stats']['mean_ms'] / 1000,
    ]

    x2 = np.arange(len(metrics))
    bars3 = ax2.bar(x2 - width/2, bl_vals, width, label='Baseline (K=1)',
                    color=COLORS['baseline'], alpha=0.85, edgecolor='white')
    bars4 = ax2.bar(x2 + width/2, imp_vals, width, label='K=5',
                    color=COLORS['k5'], alpha=0.85, edgecolor='white')
    ax2.set_title('(b) Hard-Case Key Metrics')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics, fontsize=8)
    ax2.legend(fontsize=7.5, frameon=True, fancybox=False, edgecolor='gray',
               loc='upper left')
    # Value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            h = bar.get_height()
            fmt = f'{h:.1f}' if h < 100 else f'{h:.0f}'
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.3, fmt,
                     ha='center', va='bottom', fontsize=7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('Hard-Case Analysis: 20-Move-Deep Scrambles (49 solved)',
                 fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig('figures/fig_hard_cases.pdf')
    fig.savefig('figures/fig_hard_cases.png')
    plt.close(fig)
    print('  ✓ fig_hard_cases.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating figures...')
    fig_move_distribution()
    fig_node_expansion()
    fig_anytime_curve()
    fig_neural_comparison()
    fig_timing_distribution()
    fig_k_scaling()
    fig_multi_seed()
    fig_hard_cases()
    print(f'\nAll figures saved to figures/')
    print('Include in LaTeX with: \\includegraphics[width=\\columnwidth]{{figures/fig_XXX.pdf}}')
