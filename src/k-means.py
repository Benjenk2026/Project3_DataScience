"""
k-means.py
Run K-Means clustering on the cleaned HIGGS dataset using all 28 feature dimensions.

Usage examples:
	python src/k-means.py
	python src/k-means.py --algorithm minibatch --rows 500000
	python src/k-means.py --benchmark-sizes 50000,100000,250000,500000
	python src/k-means.py --justify-subsampling
	python src/k-means.py --create-subsample --subsample-size 200000
"""

import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [f"feature_{i}" for i in range(1, 29)]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Cluster the HIGGS dataset with K-Means using all 28 feature columns."
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=Path("data/processed/higgs_cleaned.csv"),
		help="Path to cleaned HIGGS CSV (default: data/processed/higgs_cleaned.csv)",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("data/processed/higgs_clustered.csv"),
		help="Output CSV with assigned cluster labels.",
	)
	parser.add_argument("--k", type=int, default=2, help="Number of clusters (default: 2)")
	parser.add_argument(
		"--algorithm",
		choices=["kmeans", "minibatch"],
		default="kmeans",
		help="Use standard KMeans or MiniBatchKMeans (default: kmeans)",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=10_000,
		help="Mini-batch size when --algorithm minibatch is selected.",
	)
	parser.add_argument(
		"--random-state",
		type=int,
		default=42,
		help="Random seed for reproducibility.",
	)
	parser.add_argument(
		"--rows",
		type=int,
		default=None,
		help="Number of rows to use from the input file (default: all rows).",
	)
	parser.add_argument(
		"--benchmark-sizes",
		type=str,
		default=None,
		help="Comma-separated row counts for runtime benchmark, e.g. 50000,100000,250000.",
	)
	parser.add_argument(
		"--plot-output",
		type=Path,
		default=Path("Analysis_and_Findings/runtime_vs_size.png"),
		help="Where to save runtime-vs-size plot in benchmark mode.",
	)
	parser.add_argument(
		"--justify-subsampling",
		action="store_true",
		help="Run fixed 50k/100k/200k benchmark and save justification plot.",
	)
	parser.add_argument(
		"--justify-plot-output",
		type=Path,
		default=Path("Analysis_and_Findings/subsample_justification.png"),
		help="Where to save the fixed-size subsampling justification plot.",
	)
	parser.add_argument(
		"--create-subsample",
		action="store_true",
		help="Create stratified subsample from input and save it to disk.",
	)
	parser.add_argument(
		"--subsample-size",
		type=int,
		default=200_000,
		help="Rows to keep when --create-subsample is used (default: 200000).",
	)
	parser.add_argument(
		"--subsample-seed",
		type=int,
		default=42,
		help="Random seed for stratified subsampling (default: 42).",
	)
	parser.add_argument(
		"--subsample-output",
		type=Path,
		default=Path("data/processed/higgs_200k.csv"),
		help="Output CSV for --create-subsample mode.",
	)
	return parser.parse_args()


def validate_input(df: pd.DataFrame) -> None:
	missing = [col for col in FEATURE_COLS if col not in df.columns]
	if missing:
		raise ValueError(f"Input file is missing required feature columns: {missing}")


def build_model(args: argparse.Namespace):
	if args.algorithm == "kmeans":
		return KMeans(
			n_clusters=args.k,
			n_init=10,
			random_state=args.random_state,
		)
	return MiniBatchKMeans(
		n_clusters=args.k,
		n_init=10,
		random_state=args.random_state,
		batch_size=args.batch_size,
	)


def parse_benchmark_sizes(raw: str) -> list[int]:
	sizes = []
	for token in raw.split(","):
		value = token.strip()
		if not value:
			continue
		parsed = int(value)
		if parsed <= 0:
			raise ValueError("All benchmark sizes must be positive integers.")
		sizes.append(parsed)
	if not sizes:
		raise ValueError("--benchmark-sizes must contain at least one value.")
	return sorted(set(sizes))


def prepare_numeric_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	validate_input(df)
	X = df[FEATURE_COLS].copy().apply(pd.to_numeric, errors="coerce")
	valid_mask = X.notna().all(axis=1)
	dropped = int((~valid_mask).sum())
	if dropped:
		print(f"Dropped {dropped:,} rows with missing/non-numeric feature values.")
	return df.loc[valid_mask].copy(), X.loc[valid_mask]


def benchmark_kmeans_runtime(X_subset: pd.DataFrame, repeats: int = 3) -> tuple[float, list[float]]:
	"""
	Measure k-Means runtime on a fixed feature subset.

	Runs one untimed warm-up pass to avoid counting one-time sklearn/BLAS startup
	overhead, then reports the median across repeated timed runs.
	"""
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X_subset)
	KMeans(n_clusters=2, n_init=10, random_state=42).fit(X_scaled)

	runs = []
	for _ in range(repeats):
		start = perf_counter()
		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(X_subset)
		model = KMeans(n_clusters=2, n_init=10, random_state=42)
		model.fit(X_scaled)
		runs.append(perf_counter() - start)

	return median(runs), runs


def run_single_clustering(args: argparse.Namespace, df: pd.DataFrame) -> None:
	df, X = prepare_numeric_features(df)

	print("Standardizing all 28 feature columns...")
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	model = build_model(args)
	print(
		f"Fitting {model.__class__.__name__} with k={args.k} on "
		f"{X_scaled.shape[0]:,} rows x {X_scaled.shape[1]} features..."
	)
	cluster_labels = model.fit_predict(X_scaled)

	df["cluster"] = cluster_labels.astype("int32")
	args.output.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(args.output, index=False)

	counts = df["cluster"].value_counts().sort_index()
	print(f"Saved clustered data to: {args.output}")
	print("Cluster counts:")
	for cluster_id, count in counts.items():
		print(f"  cluster {cluster_id}: {count:,}")


def subsample_data(df: pd.DataFrame, n: int = 200000, seed: int = 42) -> pd.DataFrame:
	"""
	Stratified random sample from the full dataset, preserving label distribution.
	
	Args:
		df: Input DataFrame with a 'label' column
		n: Number of rows to sample (default: 200000)
		seed: Random state for reproducibility (default: 42)
	
	Returns:
		Subsampled DataFrame with stratification maintained
	"""
	if "label" not in df.columns:
		raise ValueError("DataFrame must contain a 'label' column for stratified sampling")
	
	if n >= len(df):
		print(f"Warning: requested sample size {n:,} >= dataset size {len(df):,}. Returning full dataset.")
		return df.copy()
	
	# Calculate samples per label to maintain proportion
	label_counts = df["label"].value_counts()
	total_rows = len(df)
	
	# Stratified sample preserving label proportions
	indices = []
	for label_val, count in label_counts.items():
		# Calculate how many samples for this label
		n_per_label = max(1, int(n * count / total_rows))
		label_df = df[df["label"] == label_val]
		label_indices = label_df.sample(n=min(n_per_label, len(label_df)), random_state=seed).index
		indices.extend(label_indices)
	
	subsampled = df.loc[indices].copy()
	
	# Ensure exact size by adjusting if necessary
	if len(subsampled) != n:
		if len(subsampled) > n:
			subsampled = subsampled.sample(n=n, random_state=seed)
		else:
			# Add more samples if needed
			needed = n - len(subsampled)
			remaining = df[~df.index.isin(subsampled.index)]
			extra = remaining.sample(n=min(needed, len(remaining)), random_state=seed)
			subsampled = pd.concat([subsampled, extra], ignore_index=False)
	
	subsampled = subsampled.reset_index(drop=True)
	print(f"Created stratified subsample: {len(subsampled):,} rows")
	print(f"Label distribution:\n{subsampled['label'].value_counts(normalize=True).sort_index()}")
	return subsampled


def save_processed_data(df_sub: pd.DataFrame, output_path: Path = Path("data/processed/higgs_200k.csv")) -> None:
	"""
	Save cleaned subsample to disk for reproducibility.
	
	Args:
		df_sub: Subsampled DataFrame to save
		output_path: Where to save the CSV file (default: data/processed/higgs_200k.csv)
	"""
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	df_sub.to_csv(output_path, index=False)
	print(f"Saved subsample to: {output_path}")


def justify_subsampling(
	input_path: Path = Path("data/processed/higgs_cleaned.csv"),
	output_path: Path = Path("Analysis_and_Findings/subsample_justification.png"),
) -> None:
	"""
	Benchmark k-Means on 50k/100k/200k/1M rows to justify use of subsampled dataset.
	Plots runtime vs dataset size and saves to Analysis_and_Findings/subsample_justification.png
	"""
	input_path = Path(input_path)
	output_path = Path(output_path)
	if not input_path.exists():
		raise FileNotFoundError(f"Input file not found: {input_path}")
	
	# Load only what we need for the largest benchmark
	benchmark_sizes = [50_000, 100_000, 200_000, 1_000_000]
	max_size = max(benchmark_sizes)
	
	print(f"Loading {max_size:,} rows for benchmarking...")
	df = pd.read_csv(input_path, low_memory=False, nrows=max_size)
	df, X = prepare_numeric_features(df)
	
	available_rows = len(X)
	
	# Check which sizes are feasible
	usable_sizes = [size for size in benchmark_sizes if size <= available_rows]
	skipped_sizes = [size for size in benchmark_sizes if size > available_rows]
	
	if skipped_sizes:
		print(f"Skipping sizes larger than available rows ({available_rows:,}): {skipped_sizes}")
	
	if not usable_sizes:
		raise ValueError(f"No benchmark sizes <= available rows ({available_rows:,})")
	
	print(f"\nBenchmarking k-Means on {len(usable_sizes)} dataset sizes...")
	print(
		f"Using contiguous prefixes from {input_path} after numeric validation: "
		f"rows 1..N of the cleaned feature matrix."
	)
	runtimes = []
	raw_timings = {}
	
	for size in usable_sizes:
		X_subset = X.iloc[:size]
		
		print(f"\n  Size {size:,} rows (using validated rows 1..{size:,})...")
		elapsed, runs = benchmark_kmeans_runtime(X_subset)
		runtimes.append(elapsed)
		raw_timings[size] = runs
		formatted_runs = ", ".join(f"{run:.3f}s" for run in runs)
		print(f"    Timed runs: {formatted_runs}")
		print(f"    Median runtime: {elapsed:.3f} seconds")
	
	# Plot and save
	output_path.parent.mkdir(parents=True, exist_ok=True)
	
	plt.figure(figsize=(10, 6))
	plt.plot(usable_sizes, runtimes, marker="o", linewidth=2, markersize=8, color="steelblue")
	plt.title("k-Means Runtime vs Dataset Size\n50k, 100k, 200k, and 1M Rows", fontsize=14, fontweight="bold")
	plt.xlabel("Dataset Size (rows)", fontsize=12)
	plt.ylabel("Median runtime over 3 timed runs (seconds)", fontsize=12)
	plt.grid(True, alpha=0.3)
	
	# Add value labels on points
	for size, runtime in zip(usable_sizes, runtimes):
		plt.annotate(f"{runtime:.2f}s", xy=(size, runtime), xytext=(0, 10),
					textcoords="offset points", ha="center", fontsize=10)
	
	plt.tight_layout()
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close()
	
	print(f"\n✓ Saved benchmark plot to: {output_path}")
	print("\nSummary:")
	for size, runtime in zip(usable_sizes, runtimes):
		formatted_runs = ", ".join(f"{run:.3f}s" for run in raw_timings[size])
		print(f"  {size:,} rows: median {runtime:.3f}s from [{formatted_runs}]")


def run_runtime_benchmark(args: argparse.Namespace, df: pd.DataFrame) -> None:
	sizes = parse_benchmark_sizes(args.benchmark_sizes)

	df, X = prepare_numeric_features(df)
	available_rows = len(X)
	if available_rows == 0:
		raise ValueError("No valid rows available for benchmarking.")

	usable_sizes = [size for size in sizes if size <= available_rows]
	skipped_sizes = [size for size in sizes if size > available_rows]
	if skipped_sizes:
		print(f"Skipping sizes larger than available rows ({available_rows:,}): {skipped_sizes}")
	if not usable_sizes:
		raise ValueError("No benchmark sizes are <= available valid rows.")

	runtimes = []
	for size in usable_sizes:
		X_subset = X.iloc[:size]
		start = perf_counter()
		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(X_subset)
		model = build_model(args)
		model.fit(X_scaled)
		elapsed = perf_counter() - start
		runtimes.append(elapsed)
		print(f"Size {size:,}: {elapsed:.3f} seconds")

	args.plot_output.parent.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(8, 5))
	plt.plot(usable_sizes, runtimes, marker="o", linewidth=2)
	plt.title(f"Runtime vs Dataset Size ({args.algorithm}, k={args.k})")
	plt.xlabel("Rows used")
	plt.ylabel("Runtime (seconds)")
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(args.plot_output, dpi=150)
	plt.close()

	print(f"Saved runtime plot to: {args.plot_output}")


def main() -> None:
	args = parse_args()

	if not args.input.exists():
		raise FileNotFoundError(f"Input file not found: {args.input}")

	if args.subsample_size <= 0:
		raise ValueError("--subsample-size must be a positive integer.")

	if args.justify_subsampling:
		justify_subsampling(input_path=args.input, output_path=args.justify_plot_output)
		if not args.create_subsample:
			return

	if args.create_subsample:
		print(f"Loading: {args.input}")
		df = pd.read_csv(args.input, low_memory=False)
		df_sub = subsample_data(df, n=args.subsample_size, seed=args.subsample_seed)
		save_processed_data(df_sub, output_path=args.subsample_output)
		return

	print(f"Loading: {args.input}")
	if args.benchmark_sizes:
		max_rows = max(parse_benchmark_sizes(args.benchmark_sizes))
		if args.rows is not None:
			max_rows = min(max_rows, args.rows)
		df = pd.read_csv(args.input, low_memory=False, nrows=max_rows)
		run_runtime_benchmark(args, df)
		return

	if args.rows is not None:
		if args.rows <= 0:
			raise ValueError("--rows must be a positive integer.")
		df = pd.read_csv(args.input, low_memory=False, nrows=args.rows)
	else:
		df = pd.read_csv(args.input, low_memory=False)

	run_single_clustering(args, df)


if __name__ == "__main__":
	main()
