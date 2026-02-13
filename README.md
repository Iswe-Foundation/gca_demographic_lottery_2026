# Demographic Lottery System - Code Documentation

## Overview

This code implements a **demographic lottery system** that selects a representative sample from a multiple pools of candidates corresponding to multiple map points selected during a prior [geographic lottery phase](https://github.com/Iswe-Foundation/gca_location_lottery_2025), while matching predefined demographic benchmarks. See [this website](https://globalassemblies.org/civic-assembly) for details of the broader process.

The system uses random sampling combined with a greedy swap optimization algorithm to minimize deviations from target demographic proportions across multiple criteria (e.g., gender, age, education, settlement type). It runs the sample multiple times and selects the result with the least deviation from the stratification criteria.

## Key Components

### 1. Setup and Configuration

**Paths and Constants:**
- Sets up input/output directories relative to the script location
- Defines `num_map_points` (105) - the number of geographic locations to select from
- Configures `num_runs` (20) - how many lottery simulations to perform

**Data Loading:**
- Reads candidate pool from Excel file: `CiA_demographic_lottery_2026_105.xlsx`
- Reads benchmark targets from Excel file: `Stratification_metrics.xlsx` (sheet: `computable_format`)
- See file input_formats.xlsx for the structure of these.
- There is a lot of prior work to be done by humans to recruit a demographically diverse pool at each location.

**Data Cleaning:**
- Handles a special case where pandas misinterprets "7-12" (education years) as a date
- Converts any datetime objects in the `education_years` column back to the string "7-12"

### 2. Benchmark Processing

**`convert_records_to_nested_dict()`:**
- Converts benchmark data from a DataFrame format (Criterion, Option, Proportion) to a nested dictionary
- Structure: `{criterion: {option: proportion}}`
- Example: `{'gender': {'man': 0.495, 'woman': 0.495, 'other': 0.01}}`
- Validates that all benchmark proportions sum to 1.0 for each category

### 3. Core Functions

#### `generate_single_set_of_results(pool)`
Generates one random lottery result:
1. Shuffles the pool randomly
2. Creates a results DataFrame with `map_point` values 1 to `num_results`
3. For each map point, randomly selects one person from the pool who has that same map point
4. Returns the selected results DataFrame

**Key Constraint:** Only people with the same `map_point` can be swapped, ensuring geographic/logistical feasibility.

#### `convert_results_df_to_descriptive_dict(results, benchmark)`
Converts the results DataFrame into the same nested dictionary structure as benchmarks:
- Calculates proportions for each criterion/option combination
- Returns a dictionary matching the benchmark structure

#### `disaggregated_deviation_score(results, benchmark)`
Calculates how much the results deviate from benchmarks:
- Converts results to dictionary format
- Subtracts benchmark values from result values
- Returns deviation scores (positive = surplus, negative = deficit)

#### `get_abs_errors(score)`
Aggregates absolute errors by category:
- Sums the absolute values of all deviations within each category
- Returns a single-layer dictionary: `{category: total_abs_error}`

#### `total_abs_error(abs_errors)`
Calculates the overall total absolute error across all categories.

#### `find_worst_deviation(scores)`
Identifies the largest deviation (in absolute terms) across all categories and options:
- Returns: `(worst_category, worst_option, worst_deviation)`
- Used by the optimization algorithm to prioritize improvements

### 4. Greedy Swap Optimization

**`greedy_swap_improvement(results, pool, benchmark, max_iterations, min_improvement)`**

This is the core optimization algorithm that improves lottery results by swapping candidates:

**Algorithm Flow:**
1. **Initialize:** Create a lookup dictionary grouping pool candidates by `map_point`
2. **Calculate Current Error:** Compute deviation scores and total absolute error
3. **Iterative Improvement Loop** (up to `max_iterations`):
   - Find the worst deviation (largest absolute error)
   - If deviation is below `min_improvement` threshold, stop
   - For each person in current results:
     - Get candidates from the same `map_point` (logistical constraint)
     - Filter candidates that could improve the worst deviation:
       - If worst deviation is negative (deficit), look for candidates with the needed option
       - If worst deviation is positive (surplus), look for candidates without that option
     - Test each valid candidate swap:
       - Calculate new error if swap is made
       - Track the best improvement found
   - Apply the best swap found
   - Update error calculations
   - Repeat until no improvement or max iterations reached

**Key Features:**
- Only swaps people with the same `map_point` (maintains geographic constraints)
- Greedy approach: always makes the best available swap at each step
- Stops early if no meaningful improvement is possible
- Returns comprehensive summary including optimized results, scores, errors, and iteration count

### 5. Main Execution Loop

The script runs multiple lottery simulations:

```python
for run_number in range(num_runs):
    1. Generate random initial selection
    2. Optionally run greedy swap optimization
    3. Calculate deviation scores and errors
    4. Store results in all_runs list
    5. Print progress (iterations used, time taken)
```

**Configuration:**
- `use_greedy_swaps = True` - Enable/disable optimization
- `greedy_swap_iterations = 300` - Maximum swaps per run
- `greedy_swap_min_improvement = 1e-6` - Minimum improvement threshold

### 6. Best Run Selection

**`find_best_run(all_runs)`:**
- Identifies the run with the lowest total absolute error
- This represents the selection that best matches the demographic benchmarks

### 7. Visualization

Creates horizontal bar charts showing:
- Deviation from benchmark (in percentage) for each option within each category
- Positive values = surplus, negative values = deficit
- All charts use the same x-axis scale for easy comparison
- Title shows which run number was selected as best

### 8. Output Files

**Text Summary (`deviations_YYYY_MM_DD_at_HH_MM_SS.txt`):**
- Detailed deviations by category and option
- Both percentage and decimal formats

**CSV Results (`demographic_lotteryresults_YYYY_MM_DD_at_HH_MM_SS.csv`):**
- Complete results DataFrame for the best run
- Contains all demographic information for selected candidates

## Algorithm Summary

1. **Random Initialization:** Generate random selection respecting map_point constraints
2. **Optimization (Optional):** Use greedy swaps to minimize demographic deviations
3. **Evaluation:** Calculate how well results match benchmarks
4. **Selection:** Choose the best run from multiple simulations
5. **Export:** Save results and visualizations

## Key Design Principles

- **Geographic Constraint:** All swaps maintain `map_point` matching (logistical feasibility)
- **Multi-Criteria Optimization:** Simultaneously optimizes across multiple demographic dimensions
- **Stochastic Approach:** Runs multiple simulations and selects the best outcome
- **Greedy Local Search:** Makes locally optimal swaps to improve global fit

## Performance Considerations

- The greedy swap algorithm has O(n²) complexity per iteration (where n = num_results)
- Multiple runs allow exploration of solution space
- Early stopping when improvements become negligible
- Progress tracking shows time per run and total execution time
