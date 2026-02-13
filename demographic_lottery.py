# This code is for running the demographic lottery - the 2nd part of a 2 stage selection process for the 2026 Global Citizens' Assembly's Civic Assembly
# Requires a local pool of potential Assembly Members for each of the map points, and a benchmark of the desired demographic proportions for each of the stratification criteria
# Released under MIT license
# By JSD, Dec '25 - Jan '26

#%%
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import make_dummy_data
import importlib
from pathlib import Path
import json
from datetime import datetime

# Get the script's directory and build paths relative to it
script_dir = Path(__file__).parent
out_path = script_dir / "outputs"
in_path = script_dir / "resources"

#define key constants
num_map_points=105 
num_runs = 50

# read the all_pools_dummy_data.csv file
#pool = pd.read_csv(in_path / 'all_pools_dummy_data.csv')

# generate dummy data
#num_at_each_map_point = random.randint(4, 10)
#importlib.reload(make_dummy_data) 
#pool = make_dummy_data.generate_dummy_data(num_map_points=num_map_points, num_at_each_map_point=num_at_each_map_point)



# read the real pool data
# Read the file - pandas may convert "7-12" to datetime, we'll fix it afterwards
pool = pd.read_excel(in_path / 'CiA_demographic_lottery_2026_105.xlsx')

# DEBUG: Check what we got after reading
# print("=" * 60)
# print("DEBUG: After reading Excel file")
# if 'education_years' in pool.columns:
#     print(f"Column dtype: {pool['education_years'].dtype}")
#     print(f"Sample values (first 10):")
#     for idx in pool.index[:10]:
#         val = pool.at[idx, 'education_years']
#         print(f"  Index {idx}: {repr(val)} (type: {type(val).__name__})")
# print("=" * 60)

# Fix education_years column if it contains datetime objects
# pandas often interprets "7-12" as a date and converts it to a datetime
# Since "7-12" is the only ambiguous value, convert all datetimes to "7-12"
if 'education_years' in pool.columns:
    # Use a for loop to directly assign values, bypassing any pandas type inference
    for idx in pool.index:
        val = pool.at[idx, 'education_years']
        
        if pd.isna(val):
            continue
        
        # Check if it's a datetime/Timestamp object (pandas Timestamp or Python datetime)
        if isinstance(val, (pd.Timestamp, datetime)):
            pool.at[idx, 'education_years'] = "7-12"
        # Check if it's a string that looks like a datetime
        elif isinstance(val, str):
            # Check for explicit datetime string format
            if val == "2026-12-07 00:00:00" or "2026-12-07" in val:
                pool.at[idx, 'education_years'] = "7-12"
            # Check if it looks like a datetime string in general
            elif len(val) > 10 and ('202' in val or '200' in val) and ('-' in val or '/' in val):
                try:
                    # Try to parse as datetime - if it succeeds, it's a datetime string
                    pd.to_datetime(val)
                    pool.at[idx, 'education_years'] = "7-12"
                except:
                    # Not a parseable datetime, leave as is
                    pass
    
    # print(f"DEBUG: Total conversions made: {conversion_count}")
    # print("=" * 60)
    
    # DEBUG: Check what we have after conversion
    # print("DEBUG: After conversion")
    # print(f"Column dtype: {pool['education_years'].dtype}")
    # print(f"Sample values (first 10):")
    # for idx in pool.index[:10]:
    #     val = pool.at[idx, 'education_years']
    #     print(f"  Index {idx}: {repr(val)} (type: {type(val).__name__})")
    # print("=" * 60)

def convert_records_to_nested_dict(records_df):
    """
    Convert a DataFrame with columns 'Criterion', 'Option', 'Proportion' 
    to a nested dictionary format: {criterion: {option: proportion}}
    
    Args:
        records_df: DataFrame with columns 'Criterion', 'Option', 'Proportion'
    
    Returns:
        Nested dictionary in format {criterion: {option: proportion}}
    """
    nested_dict = {}
    
    # Normalize column names to handle case variations
    col_map = {}
    for col in records_df.columns:
        col_lower = col.lower()
        if 'criterion' in col_lower:
            col_map['criterion'] = col
        elif 'option' in col_lower:
            col_map['option'] = col
        elif 'proportion' in col_lower or 'value' in col_lower:
            col_map['proportion'] = col
    
    if len(col_map) != 3:
        raise ValueError(f"Expected columns 'Criterion', 'Option', 'Proportion'. Found: {list(records_df.columns)}")
    
    criterion_col = col_map['criterion']
    option_col = col_map['option']
    proportion_col = col_map['proportion']
    
    # Group by criterion and build nested structure
    for criterion, group in records_df.groupby(criterion_col):
        nested_dict[criterion] = dict(zip(group[option_col], group[proportion_col]))
    
    return nested_dict


#read in the real benchmarks
benchmarks_data = pd.read_excel(in_path / 'Stratification_metrics.xlsx', sheet_name='computable_format')
print("Raw benchmarks data:")
print(benchmarks_data.head())

#convert the benchmarks_data dataframe to nested dictionary format
benchmark = convert_records_to_nested_dict(benchmarks_data)
print("\nConverted to nested dictionary format:")
print(json.dumps(benchmark, indent=4))

#%%

# OLD HARDCODED BENCHMARK (replaced by Excel import above)
"""
benchmark = {
    'gender': {
        'man': 0.495,
        'woman': 0.495,
        'other': 0.01
    },
    'age_group': {
        'under 25': 0.25,
        '25-39': 0.25,
        '40-59': 0.25,
        '60+': 0.25
    },
    'education_years': {
        '0-6': 0.45,
        '7-11': 0.45,
        '12+': 0.10
    },
    'settlement_type': {
        'urban center': 0.3,
        'urban cluster': 0.3,
        'rural': 0.4
    },
    'topic_question': {
        'a': 0.15,
        'b': 0.20,
        'c': 0.20,
        'd': 0.20,
        'e': 0.20,
        'f': 0.05
    }
}
"""
#define the number of results to generate
num_results = num_map_points

# Check what map_points are available in the pool and warn if there's a mismatch
if 'map_point' in pool.columns:
    available_map_points = sorted(pool['map_point'].unique())
    print(f"Available map_points in pool: {min(available_map_points)} to {max(available_map_points)}")
    print(f"Total unique map_points: {len(available_map_points)}")
    print(f"Requested num_results: {num_results}")
    if num_results > len(available_map_points):
        print(f"WARNING: num_results ({num_results}) is greater than available map_points ({len(available_map_points)})")
        print("This will cause an error when trying to sample. Consider reducing num_results or checking your data.")

# define the number of times to run the lottery
num_runs = 20

# DEBUG: Final check before lottery runs
# print("=" * 60)
# print("DEBUG: Final check before lottery runs")
# if 'education_years' in pool.columns:
#     print(f"Column dtype: {pool['education_years'].dtype}")
#     # Check for any datetime values
#     datetime_count = 0
#     for idx in pool.index:
#         val = pool.at[idx, 'education_years']
#         if isinstance(val, (pd.Timestamp, datetime)):
#             datetime_count += 1
#             print(f"  WARNING: Found datetime at index {idx}: {val} (type: {type(val).__name__})")
#         elif isinstance(val, str) and ('2026-12-07' in val or '2026-07-12' in val):
#             datetime_count += 1
#             print(f"  WARNING: Found datetime string at index {idx}: {repr(val)}")
#     if datetime_count == 0:
#         print("  ✓ No datetime values found - all good!")
#     else:
#         print(f"  ✗ Found {datetime_count} datetime values that need fixing!")
# print("=" * 60)

# toggle greedy swap optimisation
use_greedy_swaps = True
greedy_swap_iterations = 300
greedy_swap_min_improvement = 1e-6

#checksum each category to ensure it adds up to 1
for category, values in benchmark.items():
    total = sum(values.values())
    if total != 1:
        print(f"Error: {category} does not add up to 1")
        print(f"Total: {total}")
        print(f"Values: {values}")
        exit()


def empty_like(template):
    if isinstance(template, dict):
        return {key: empty_like(value) for key, value in template.items()}
    return 0.0 
    
def generate_single_set_of_results(pool):
    results = pd.DataFrame(columns=pool.columns)
    # randomly shuffle the pool
    pool = pool.sample(frac=1).reset_index(drop=True)
    # fill the map_point column with the numbers 1 to num_results
    results['map_point'] = range(1, num_results + 1)
    # for each row, randomly pick someone from the pool who has the same map_point as the row
    for index, row in results.iterrows():
        # randomly pick someone from the pool who has the same map_point as the row
        random_person = pool[pool['map_point'] == row['map_point']].sample(n=1)
        # add the random person to the results dataframe
        # Use iloc[0] to get a Series, which pandas will align by column name automatically
        results.loc[index] = random_person.iloc[0]
    return results


def subtract_dicts(dict1, dict2):
    # subtract dict2 from dict1 recursively
    result = {}
    for key in dict1:
        if isinstance(dict1[key], dict):
            result[key] = subtract_dicts(dict1[key], dict2[key])
        else:
            result[key] = dict1[key] - dict2[key]
    return result





def convert_results_df_to_descriptive_dict(results, benchmark):
    # create a descriptive dictionary from the results dataframe
    #first make a copy of benchmark to get the structure
    results_dict = empty_like(benchmark)
    
    # Only process columns that exist in the benchmark (skip metadata columns like map_point, person_number, name)
    for criterion in benchmark.keys():
        if criterion in results.columns:
            proportions = results[criterion].value_counts(normalize=True).to_dict()
            for option in proportions:
                results_dict[criterion][option] = proportions[option]
            
    #print('results dict', results_dict)
    return results_dict

def disaggregated_deviation_score(results, benchmark):
    #convert the results dataframe to a dictionary
    results_dict = convert_results_df_to_descriptive_dict(results, benchmark)

    #print('results dict', results_dict)
    scores = subtract_dicts(results_dict, benchmark)
    return scores

def get_abs_errors(score):
    #add up the absolute values for each category and return a 1 layer dict with the total abs error for each category
    abs_errors = {}
    for category, values in score.items():
        abs_errors[category] = sum(abs(value) for value in values.values())
    #print('abs_errors', abs_errors)
    return abs_errors


def total_abs_error(abs_errors):
    return sum(abs_errors.values())


def find_worst_deviation(scores):
    worst_category = None
    worst_option = None
    worst_deviation = 0.0
    for category, options in scores.items():
        for option, deviation in options.items():
            if abs(deviation) > abs(worst_deviation):
                worst_deviation = deviation
                worst_category = category
                worst_option = option
    return worst_category, worst_option, worst_deviation


def greedy_swap_improvement(results, pool, benchmark, max_iterations=500, min_improvement=1e-6):
    """Perform greedy swap optimisation on a set of results. Swaps are limited to people
    sharing the same `map_point`, ensuring the logistics of replacements remain feasible."""
    results = results.copy().reset_index(drop=True)
    pool_lookup = {map_point: group.reset_index(drop=True)
                   for map_point, group in pool.groupby('map_point')}

    results_dict = convert_results_df_to_descriptive_dict(results, benchmark)
    scores = subtract_dicts(results_dict, benchmark)
    abs_errors = get_abs_errors(scores)
    current_total_error = total_abs_error(abs_errors)

    iterations_used = 0
    for _ in range(max_iterations):
        worst_category, worst_option, worst_deviation = find_worst_deviation(scores)
        if worst_category is None or abs(worst_deviation) <= min_improvement:
            break

        best_candidate_data = None
        best_candidate_total_error = current_total_error

        for idx, row in results.iterrows():
            map_point = row['map_point']
            candidate_group = pool_lookup.get(map_point)
            if candidate_group is None:
                continue

            if worst_deviation < 0 and row[worst_category] == worst_option:
                # we already have the needed option, swapping it out would make it worse
                continue

            if worst_deviation > 0 and row[worst_category] != worst_option:
                # this row is not contributing to the surplus; swapping it won't help
                continue

            for _, candidate in candidate_group.iterrows():
                if worst_deviation < 0 and candidate[worst_category] != worst_option:
                    continue
                if worst_deviation > 0 and candidate[worst_category] == worst_option:
                    continue

                if candidate.equals(row):
                    continue

                new_results = results.copy()
                # Select columns by name to ensure correct alignment even if columns are in different order
                # candidate is already a Series from iterrows(), so selecting columns returns a Series
                candidate_series = candidate[new_results.columns]
                new_results.loc[idx] = candidate_series

                new_results_dict = convert_results_df_to_descriptive_dict(new_results, benchmark)
                new_scores = subtract_dicts(new_results_dict, benchmark)
                new_abs_errors = get_abs_errors(new_scores)
                new_total_error = total_abs_error(new_abs_errors)

                if new_total_error < best_candidate_total_error - min_improvement:
                    best_candidate_total_error = new_total_error
                    best_candidate_data = (
                        idx,
                        candidate_series,
                        new_results_dict,
                        new_scores,
                        new_abs_errors
                    )

        if best_candidate_data is None:
            break

        idx, candidate_values, new_results_dict, new_scores, new_abs_errors = best_candidate_data
        results.loc[idx] = candidate_values
        results_dict = new_results_dict
        scores = new_scores
        abs_errors = new_abs_errors
        current_total_error = best_candidate_total_error
        iterations_used += 1
        #print(f'iteration {iterations_used} of {max_iterations} - total error: {current_total_error}')

    return {
        'results': results,
        'results_dict': results_dict,
        'scores': scores,
        'abs_errors': abs_errors,
        'total_abs_error': current_total_error,
        'iterations': iterations_used
    }



all_runs = []

# save start time
start_time = time.time()
print(f'Start time: {start_time}')

"""
# run the lottery num_runs times
for run_number in range(num_runs):
    
    #print('run_number', run_number)
    results = generate_single_set_of_results(pool)
    optimisation_summary = None

    if use_greedy_swaps:
        optimisation_summary = greedy_swap_improvement(
            results,
            pool,
            benchmark,
            max_iterations=greedy_swap_iterations,
            min_improvement=greedy_swap_min_improvement
        )
    """

for run_number in range(num_runs):
    this_run_start_time = time.time()
    #print('run_number', run_number)
    results = generate_single_set_of_results(pool)
    optimisation_summary = None

    if use_greedy_swaps:
        optimisation_summary = greedy_swap_improvement(
            results,
            pool,
            benchmark,
            max_iterations=greedy_swap_iterations,
            min_improvement=greedy_swap_min_improvement
        )
        results = optimisation_summary['results']
        results_dict = optimisation_summary['results_dict']
        individual_scores = optimisation_summary['scores']
        abs_errors = optimisation_summary['abs_errors']
    else:
        results_dict = convert_results_df_to_descriptive_dict(results, benchmark)
        individual_scores = subtract_dicts(results_dict, benchmark)
        abs_errors = get_abs_errors(individual_scores)

    total_error = total_abs_error(abs_errors)
    #print('abs_errors', abs_errors)
    all_runs.append({
        'abs_errors': abs_errors,
        'errors_by_category': individual_scores,
        'results': results,
        'total_abs_error': total_error,
        'greedy_iterations': optimisation_summary['iterations'] if optimisation_summary else 0
    })
    #print(len(all_runs))
    this_run_end_time = time.time()
    print(f'Run {run_number} made {optimisation_summary['iterations'] if optimisation_summary else 0} greedy swaps and took {this_run_end_time - this_run_start_time:.2f} seconds')

#print('all_runs', all_runs)
print(f'Total time: {time.time() - start_time:.2f} seconds')
#%%
"""
# make plots of all the scores
num_categories = len(benchmark)
cols = 3                          # tweak as you like
rows = int(np.ceil(num_categories / cols))
fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*3), squeeze=False)

for idx, (category, targets) in enumerate(benchmark.items()):
    ax = axs[idx // cols, idx % cols]
    
    # collect a list of scores over runs for every option in this category
    for option in targets:        # same keys as benchmark and score dict
        y = [run['score'][category][option] for run in all_runs]
        ax.scatter(range(len(all_runs)), y, marker='o', label=option)
    
    ax.set_title(category)
    ax.set_xlabel('Run')
    ax.set_ylabel('Deviation')
    ax.legend()

# Hide any unused subplot slots
for j in range(num_categories, rows * cols):
    fig.delaxes(axs[j // cols, j % cols])

fig.tight_layout()
plt.show()
"""

#%%

def find_best_run(all_runs):
    # find the run with the lowest total abs error
    best_run = min(all_runs, key=lambda x: sum(x['abs_errors'].values()))
    return best_run

best_run = find_best_run(all_runs)
#print(f'Number of rows in dummy dataset: {num_at_each_map_point * num_map_points}')
print(f'best absolute deviation by category from all {num_runs} runs:')
for category, values in best_run['abs_errors'].items():
    print(category, f"{values*100:.1f}%")
 
#%%


# make plots of all the scores
num_categories = len(benchmark)
cols = 1                          # tweak as you like
rows = int(np.ceil(num_categories / cols))
fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*1.5), squeeze=False)

# First, find maximum absolute deviation (in percent) across all errors
all_deviations = []
for category, targets in benchmark.items():
    for option in targets:
        y = best_run['errors_by_category'][category][option]
        all_deviations.append(abs(y*100))
max_deviation = math.ceil(max(all_deviations)) if all_deviations else 0

# Now do the plotting
for idx, (category, targets) in enumerate(benchmark.items()):
    ax = axs[idx // cols, idx % cols]
    
    for option in targets:
        y = best_run['errors_by_category'][category][option]
        ax.barh(option, y*100)
    
    #ax.set_xlabel('Deviation from Benchmark (%)')
    ax.set_ylabel(category, labelpad=5 if idx % cols == 0 else 0)
    ax.set_xlim(-max_deviation, max_deviation)
    ax.legend()
    ax.xaxis.grid(True, which='major', linestyle='-', color='#dddddd', alpha=0.7)  # Add vertical grid lines

# Hide any unused subplot slots
for j in range(num_categories, rows * cols):
    fig.delaxes(axs[j // cols, j % cols])

# Align all y-axis labels to match the position of the left-most subplot's y-label
fig.align_ylabels(axs[:, 0])

plt.suptitle(f'Deviation (%) from benchmark for best of {num_runs} runs')
fig.tight_layout()
plt.show()
# %%

# and also export a text summary of the best runs's detailed deviations by category and options
import datetime

now = datetime.datetime.now()
filename = f"deviations_{now.strftime('%Y_%m_%d_at_%H_%M_%S')}.txt"

with open(out_path / filename, "w") as f:
    f.write("Detailed deviations by category and options for the best run from benchmark (%):\n")
    for category, values in best_run['errors_by_category'].items():
        f.write(f"{category}:\n")
        for option, value in values.items():
            f.write(f"    {option}: {value*100:.2f}%\n")
    f.write("Detailed deviations by category and options for the best run from benchmark (not percent):\n")
    for category, values in best_run['errors_by_category'].items():
        f.write(f"{category}:\n")
        for option, value in values.items():
            f.write(f"    {option}: {value}\n")

print(f"Deviations summary exported to {filename}")
# %%

#export the best run's results to a csv file
now = datetime.datetime.now()
filename = f"demographic_lotteryresults_{now.strftime('%Y_%m_%d_at_%H_%M_%S')}.csv"
best_run['results'].to_csv(out_path / filename, index=False)
print(f"Best run results exported to {filename}")
# %%