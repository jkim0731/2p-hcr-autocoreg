import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_volume_bounds_from_data(landmarks_df, margin_pix=0):
    """Determine volume bounds based on actual landmark distribution."""
    bounds = {
        'x': (landmarks_df['czstack_x'].min() - margin_pix,
              landmarks_df['czstack_x'].max() + margin_pix),
        'y': (landmarks_df['czstack_y'].min() - margin_pix,
              landmarks_df['czstack_y'].max() + margin_pix),
        'z': (landmarks_df['czstack_z'].min() - margin_pix,
              landmarks_df['czstack_z'].max() + margin_pix)
    }
    return bounds


def _sample_by_grid_exact(
    region_df,
    x_edges,
    y_edges,
    z_edges,
    keep_proportion,
    min_landmarks_per_grid=1,
    random_state=42,
):
    """
    Sample near target count per region while preserving grid spread.

    Fail-safe:
    - Keep at least `min_landmarks_per_grid` points in each occupied grid cell
      (or all points if the cell has fewer).
    """
    if len(region_df) == 0:
        return region_df.copy()

    keep_proportion = float(np.clip(keep_proportion, 0.0, 1.0))
    min_landmarks_per_grid = max(0, int(min_landmarks_per_grid))

    target_total = int(np.round(len(region_df) * keep_proportion))

    region = region_df.copy()
    region['grid_x'] = np.digitize(region['czstack_x'], x_edges) - 1
    region['grid_y'] = np.digitize(region['czstack_y'], y_edges) - 1
    region['grid_z'] = np.digitize(region['czstack_z'], z_edges) - 1

    grouped = list(region.groupby(['grid_x', 'grid_y', 'grid_z']))
    rng = np.random.default_rng(random_state)

    selected_indices = []

    # Pass 1: enforce fail-safe minimum per occupied grid
    for _, cell_group in grouped:
        n_base = min(len(cell_group), min_landmarks_per_grid)
        if n_base > 0:
            picks = rng.choice(cell_group.index.to_numpy(), size=n_base, replace=False)
            selected_indices.extend(picks.tolist())

    selected_indices = list(dict.fromkeys(selected_indices))

    # Pass 2: fill toward target proportion if needed
    remaining_needed = max(0, target_total - len(selected_indices))
    if remaining_needed > 0:
        remaining_pool = region.index.difference(pd.Index(selected_indices)).to_numpy()
        if len(remaining_pool) > 0:
            n_extra = min(remaining_needed, len(remaining_pool))
            extra_picks = rng.choice(remaining_pool, size=n_extra, replace=False)
            selected_indices.extend(extra_picks.tolist())

    # If minimum-per-grid exceeds target, selected count can be above target by design
    if not selected_indices:
        return region_df.iloc[0:0].copy()

    sampled = region.loc[selected_indices].drop(columns=['grid_x', 'grid_y', 'grid_z'])
    return sampled


def grid_sample_landmarks(
    landmarks_df,
    grid_size_pix=(50, 50, 50),
    edge_margin_pix=(20, 20, 30),
    interior_keep_proportion=0.1,
    edge_keep_proportion=0.2,
    min_landmarks_per_grid=1,
    random_state=42,
):
    """Grid-based sampling with separate keep proportions for interior and edge landmarks."""

    bounds = get_volume_bounds_from_data(landmarks_df, margin_pix=0)
    print("Volume bounds from data:")
    print(f"  X: {bounds['x'][0]:.0f} - {bounds['x'][1]:.0f} pixels")
    print(f"  Y: {bounds['y'][0]:.0f} - {bounds['y'][1]:.0f} pixels")
    print(f"  Z: {bounds['z'][0]:.0f} - {bounds['z'][1]:.0f} pixels\n")

    x_min, x_max = bounds['x']
    y_min, y_max = bounds['y']
    z_min, z_max = bounds['z']

    near_edge_x = (landmarks_df['czstack_x'] < x_min + edge_margin_pix[0]) | \
                  (landmarks_df['czstack_x'] > x_max - edge_margin_pix[0])
    near_edge_y = (landmarks_df['czstack_y'] < y_min + edge_margin_pix[1]) | \
                  (landmarks_df['czstack_y'] > y_max - edge_margin_pix[1])
    near_edge_z = (landmarks_df['czstack_z'] < z_min + edge_margin_pix[2]) | \
                  (landmarks_df['czstack_z'] > z_max - edge_margin_pix[2])

    near_edge = near_edge_x | near_edge_y | near_edge_z
    edge_lm = landmarks_df[near_edge]
    interior_lm = landmarks_df[~near_edge]

    print(f"Edge landmarks: {len(edge_lm)} ({100 * len(edge_lm) / len(landmarks_df):.1f}%)")
    print(f"Interior landmarks: {len(interior_lm)} ({100 * len(interior_lm) / len(landmarks_df):.1f}%)\n")

    x_edges = np.arange(x_min, x_max + grid_size_pix[0], grid_size_pix[0])
    y_edges = np.arange(y_min, y_max + grid_size_pix[1], grid_size_pix[1])
    z_edges = np.arange(z_min, z_max + grid_size_pix[2], grid_size_pix[2])

    print("Grid cells:")
    print(f"  X: {len(x_edges)-1} cells × {grid_size_pix[0]} pixels")
    print(f"  Y: {len(y_edges)-1} cells × {grid_size_pix[1]} pixels")
    print(f"  Z: {len(z_edges)-1} cells × {grid_size_pix[2]} pixels")
    print(f"  Total: {(len(x_edges)-1)*(len(y_edges)-1)*(len(z_edges)-1)} cells")
    print(f"  Min landmarks per occupied cell: {min_landmarks_per_grid}\n")

    sampled_interior_df = _sample_by_grid_exact(
        interior_lm,
        x_edges,
        y_edges,
        z_edges,
        keep_proportion=interior_keep_proportion,
        min_landmarks_per_grid=min_landmarks_per_grid,
        random_state=random_state,
    )

    sampled_edge_df = _sample_by_grid_exact(
        edge_lm,
        x_edges,
        y_edges,
        z_edges,
        keep_proportion=edge_keep_proportion,
        min_landmarks_per_grid=min_landmarks_per_grid,
        random_state=random_state + 1,
    )

    sampled_all = pd.concat([sampled_interior_df, sampled_edge_df], axis=0)
    removed = landmarks_df.loc[~landmarks_df.index.isin(sampled_all.index)]

    stats = {
        'total': len(landmarks_df),
        'edge': len(edge_lm),
        'interior': len(interior_lm),
        'sampled_interior': len(sampled_interior_df),
        'sampled_edge': len(sampled_edge_df),
        'sampled_total': len(sampled_all),
        'removed': len(removed),
        'reduction_pct': 100 * (1 - len(sampled_all) / len(landmarks_df)),
        'interior_kept_pct': 100 * len(sampled_interior_df) / max(1, len(interior_lm)),
        'edge_kept_pct': 100 * len(sampled_edge_df) / max(1, len(edge_lm)),
    }

    return {
        'sampled': sampled_all,
        'edge': edge_lm,
        'interior': interior_lm,
        'removed': removed,
        'stats': stats,
    }

def visualize_landmark_distribution(landmarks_df, sampled_df, edge_lm, interior_lm):

    # Visualize landmark distribution before/after filtering
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Use moderate parameters for visualization
    final_result = grid_sample_landmarks(
        paired_landmarks,
        grid_size_pix=(75, 75, 100),
        edge_margin_pix=(30, 30, 50),
        interior_keep_proportion=0.10,
        edge_keep_proportion=0.20
    )

    # Split kept points into interior-kept and edge-kept
    kept_df = final_result['sampled']
    edge_kept_idx = kept_df.index.intersection(final_result['edge'].index)
    edge_kept = kept_df.loc[edge_kept_idx]
    interior_kept = kept_df.drop(index=edge_kept_idx)

    # Z projection (XY plane)
    ax = axes[0, 0]
    ax.scatter(paired_landmarks['czstack_x'], paired_landmarks['czstack_y'],
            alpha=0.25, s=10, label='Original', color='gray')
    ax.scatter(interior_kept['czstack_x'], interior_kept['czstack_y'],
            alpha=0.75, s=24, label='Kept (interior)', color='blue')
    ax.scatter(edge_kept['czstack_x'], edge_kept['czstack_y'],
            alpha=0.9, s=60, label='Kept (edge)', color='blue', marker='*')
    ax.set_xlabel('Z-stack X (pixels)')
    ax.set_ylabel('Z-stack Y (pixels)')
    ax.set_title('XY Projection')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Vertical projection (YZ plane)
    ax = axes[0, 1]
    ax.scatter(paired_landmarks['czstack_y'], paired_landmarks['czstack_z'],
            alpha=0.25, s=10, label='Original', color='gray')
    ax.scatter(interior_kept['czstack_y'], interior_kept['czstack_z'],
            alpha=0.75, s=24, label='Kept (interior)', color='blue')
    ax.scatter(edge_kept['czstack_y'], edge_kept['czstack_z'],
            alpha=0.9, s=60, label='Kept (edge)', color='blue', marker='*')
    ax.set_xlabel('Z-stack Y (pixels)')
    ax.set_ylabel('Z-stack Z (pixels)')
    ax.set_title('YZ Projection')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Lateral projection (XZ plane)
    ax = axes[1, 0]
    ax.scatter(paired_landmarks['czstack_x'], paired_landmarks['czstack_z'],
            alpha=0.25, s=10, label='Original', color='gray')
    ax.scatter(interior_kept['czstack_x'], interior_kept['czstack_z'],
            alpha=0.75, s=24, label='Kept (interior)', color='blue')
    ax.scatter(edge_kept['czstack_x'], edge_kept['czstack_z'],
            alpha=0.9, s=60, label='Kept (edge)', color='blue', marker='*')
    ax.set_xlabel('Z-stack X (pixels)')
    ax.set_ylabel('Z-stack Z (pixels)')
    ax.set_title('XZ Projection')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Histogram of point density (kept points only are colored)
    ax = axes[1, 1]
    ax.hist(paired_landmarks['czstack_z'], bins=20, alpha=0.35, label='Original', color='gray')
    ax.hist(kept_df['czstack_z'], bins=20, alpha=0.8, label='Kept (all)', color='blue')
    ax.set_xlabel('Z (pixels)')
    ax.set_ylabel('Count')
    ax.set_title('Z Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()
    return fig