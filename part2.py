"""
Part 2: Performance Comparisons

In this part, we will explore comparing the performance
of different pipelines.
First, we will set up some helper classes.
Then we will do a few comparisons
between two or more versions of a pipeline
to report which one is faster.
"""

import part1

# required for timing tasks
import timeit
import time
import matplotlib.pyplot as plt
import pandas as pd

"""
=== Questions 1-5: Throughput and Latency Helpers ===

We will design and fill out two helper classes.

The first is a helper class for throughput (Q1).
The class is created by adding a series of pipelines
(via .add_pipeline(name, size, func))
where name is a title describing the pipeline,
size is the number of elements in the input dataset for the pipeline,
and func is a function that can be run on zero arguments
which runs the pipeline (like def f()).

The second is a similar helper class for latency (Q3).

1. Throughput helper class

Fill in the add_pipeline, eval_throughput, and generate_plot functions below.
"""

# Number of times to run each pipeline in the following results.
# You may modify this as you go through the file if you like, but make sure
# you set it back to 10 at the end before you submit.
NUM_RUNS = 10


class ThroughputHelper:
    def __init__(self):
        # Initialize the object.
        # Pipelines: a list of functions, where each function
        # can be run on no arguments.
        # (like: def f(): ... )
        self.pipelines = []

        # Pipeline names
        # A list of names for each pipeline
        self.names = []

        # Pipeline input sizes
        self.sizes = []

        # Pipeline throughputs
        # This is set to None, but will be set to a list after throughputs
        # are calculated.
        self.throughputs = None

    def add_pipeline(self, name, size, func):
        self.names.append(name)
        self.sizes.append(size)
        self.pipelines.append(func)

    def compare_throughput(self):
        # Measure the throughput of all pipelines
        # and store it in a list in self.throughputs.
        # Make sure to use the NUM_RUNS variable.
        # Also, return the resulting list of throughputs,
        # in **number of items per second.**

        # return list
        tput = []

        # iterate over pairs of inputsize and fps
        for input_size, funcptr in zip(self.sizes, self.pipelines):
            # we will average the throughput over NUM_RUNS trials.
            # print(f"DEBUG: {input_size}, fp: {funcptr}")

            # get running time
            timer = timeit.Timer(funcptr)
            result = timer.timeit(NUM_RUNS)

            # timeit accumulates the time, but does not avg, so we need to also divide by NUM_RUNS
            result = result / NUM_RUNS
            # Throughput is number of items processed / total running time.
            tput.append(input_size / result)

        self.throughputs = tput

    def generate_plot(self, filename):
        # Generate a plot for throughput using matplotlib.
        # You can use any plot you like, but a bar chart probably makes
        # the most sense.
        # Make sure you include a legend.
        # Save the result in the filename provided.

        # we want to include information about each set's runsize, so let's modify self.names
        names_with_time = self.sizes.copy()
        for i in range(len(names_with_time)):
            names_with_time[i] = f"{self.names[i]}\ninput size {self.sizes[i]}"

        # use a barplot
        plt.bar(names_with_time, self.throughputs)

        # configure some axis labelling, etc.
        plt.xlabel("Pipeline")
        # Set the font size here to stop graph from smushing
        for label in plt.gca().get_xticklabels():
            label.set_fontsize(5)
        plt.ylabel("Throughput (inputs / sec)")
        plt.suptitle("Comparison of Pipeline Throughput")
        plt.title(f"Retry Count: {NUM_RUNS}")
        plt.tight_layout()

        # save and clear
        plt.savefig(filename)
        plt.clf()


"""
As your answer to this part,
return the name of the method you decided to use in
matplotlib.

(Example: "boxplot" or "scatter")
"""


def q1():
    # Return plot method (as a string) from matplotlib
    return "bar"


"""
2. A simple test case

To make sure your monitor is working, test it on a very simple
pipeline that adds up the total of all elements in a list.

We will compare three versions of the pipeline depending on the
input size.
"""

LIST_SMALL = [10] * 100
LIST_MEDIUM = [10] * 100_000
LIST_LARGE = [10] * 100_000_000


def add_list(l):
    # TODO
    accumulator = 0
    for num in l:
        accumulator += num
    return accumulator


def q2a():
    # Create a ThroughputHelper object
    h = ThroughputHelper()
    # Add the 3 pipelines.
    # (You will need to create a pipeline for each one.)
    # Pipeline names: small, medium, large

    h.add_pipeline("small", len(LIST_SMALL), lambda: add_list(LIST_SMALL))
    h.add_pipeline("medium", len(LIST_MEDIUM), lambda: add_list(LIST_MEDIUM))
    h.add_pipeline("large", len(LIST_LARGE), lambda: add_list(LIST_LARGE))

    h.compare_throughput()
    h.generate_plot("output/part2-q2a.png")
    return h.throughputs
    # Generate a plot.
    # Save the plot as 'output/part2-q2a.png'.
    # TODO
    # Finally, return the throughputs as a list.
    # TODO


"""
2b.
Which pipeline has the highest throughput?
Is this what you expected?

=== ANSWER Q2b BELOW ===
The "large" pipeline had the highest throughput.
I did not expect this, since I thought that there would be more time penalty
from the OS context switching more during the longer operations
(as more simultaneous tasks will interrupt the processor than in shorter pipelines)
=== END OF Q2b ANSWER ===
"""

"""
3. Latency helper class.

Now we will create a similar helper class for latency.

The helper should assume a pipeline that only has *one* element
in the input dataset.

It should use the NUM_RUNS variable as with throughput.
"""


class LatencyHelper:
    def __init__(self):
        # Initialize the object.
        # Pipelines: a list of functions, where each function
        # can be run on no arguments.
        # (like: def f(): ... )
        self.pipelines = []

        # Pipeline names
        # A list of names for each pipeline
        self.names = []

        # Pipeline latencies
        # This is set to None, but will be set to a list after latencies
        # are calculated.
        self.latencies = None

    def add_pipeline(self, name, func):
        self.names.append(name)
        self.pipelines.append(func)

    def compare_latency(self):
        # Measure the latency of all pipelines
        # and store it in a list in self.latencies.
        # Also, return the resulting list of latencies,
        # in **milliseconds.**
        latencies = []
        for funcptr in self.pipelines:
            accumulator = 0
            for i in range(NUM_RUNS):
                start_time = time.perf_counter()
                funcptr()
                end_time = time.perf_counter()
                elapsed = (
                    end_time - start_time
                ) * 1000  # perfcounter returns in seconds.
                accumulator += elapsed
            latencies.append(accumulator / NUM_RUNS)  # get average time

        self.latencies = latencies

    def generate_plot(self, filename):
        # Generate a plot for latency using matplotlib.
        # You can use any plot you like, but a bar chart probably makes
        # the most sense.
        # Make sure you include a legend.
        # Save the result in the filename provided.

        # use a barplot again
        plt.bar(self.names, self.latencies)

        plt.xlabel("Pipeline")
        plt.ylabel("Latency (ms)")
        plt.suptitle("Comparison of Pipeline Latency")
        plt.tight_layout()

        plt.savefig(filename)
        plt.clf()


"""
As your answer to this part,
return the number of input items that each pipeline should
process if the class is used correctly.
"""


def q3():
    # Return the number of input items in each dataset,
    # for the latency helper to run correctly.

    return 1


"""
4. To make sure your monitor is working, test it on
the simple pipeline from Q2.

For latency, all three pipelines would only process
one item. Therefore instead of using
LIST_SMALL, LIST_MEDIUM, and LIST_LARGE,
for this question run the same pipeline three times
on a single list item.
"""

LIST_SINGLE_ITEM = [10]  # Note: a list with only 1 item


def q4a():
    # Create a LatencyHelper object
    h = LatencyHelper()
    # Add the single pipeline three times.
    h.add_pipeline("run1", lambda: add_list(LIST_SINGLE_ITEM))
    h.add_pipeline("run2", lambda: add_list(LIST_SINGLE_ITEM))
    h.add_pipeline("run3", lambda: add_list(LIST_SINGLE_ITEM))
    h.compare_latency()
    h.generate_plot("output/part2-q4a.png")
    # Generate a plot.
    # Save the plot as 'output/part2-q4a.png'.
    # TODO
    return h.latencies
    # Finally, return the latencies as a list.
    # TODO


"""
4b.
How much did the latency vary between the three copies of the pipeline?
Is this more or less than what you expected?

=== ANSWER Q4b BELOW ===
There is very little variance in latency, given that all latencies are all roughly sub-nanosecond timescale.
This is roughly what I expected, since the tasks all occur on the same computer, with roughly the same operating system state at all times.
=== END OF Q4b ANSWER ===
"""

"""
Now that we have our helpers, let's do a simple comparison.

NOTE: you may add other helper functions that you may find useful
as you go through this file.

5. Comparison on Part 1

Finally, use the helpers above to calculate the throughput and latency
of the pipeline in part 1.
"""

# You will need these:
# part1.load_input
# part1.PART_1_PIPELINE


def q5a():
    total_df = part1.load_input()  # since we only care about the pipeline's tp and lat
    total_size = 0
    for df in total_df:
        total_size += len(df)
    h = ThroughputHelper()

    # TODO: What do we pick as the total size?
    h.add_pipeline("part1", total_size, lambda: part1.PART_1_PIPELINE)
    h.compare_throughput()
    h.generate_plot("output/part2-q5a.png")
    return h.throughputs
    # TODO: Change this value to float if necessary.


def q5b():
    # Return the latency of the pipeline in part 1.
    part1.load_input
    h = LatencyHelper()
    h.add_pipeline("part1", part1.PART_1_PIPELINE)
    h.compare_latency()
    h.generate_plot("output/part2-q5b.png")
    return h.latencies
    # TODO: Change this value to float if necessary.


"""
===== Questions 6-10: Performance Comparison 1 =====

For our first performance comparison,
let's look at the cost of getting input from a file, vs. in an existing DataFrame.

6. We will use the same population dataset
that we used in lecture 3.

Load the data using load_input() given the file name.

- Make sure that you clean the data by removing
  continents and world data!
  (World data is listed under OWID_WRL)

Then, set up a simple pipeline that computes summary statistics
for the following:

- *Year over year increase* in population, per country

    (min, median, max, mean, and standard deviation)

How you should compute this:

- For each country, we need the maximum year and the minimum year
in the data. We should divide the population difference
over this time by the length of the time period.

- Make sure you throw out the cases where there is only one year
(if any).

- We should at this point have one data point per country.

- Finally, as your answer, return a list of the:
    min, median, max, mean, and standard deviation
  of the data.

Hints:
You can use the describe() function in Pandas to get these statistics.
You should be able to do something like
df.describe().loc["min"]["colum_name"]

to get a specific value from the describe() function.

You shouldn't use any for loops.
See if you can compute this using Pandas functions only.
"""


def load_input(filename):
    # Return a dataframe containing the population data
    # **Clean the data here**
    # TODO: Cleaning necessary?
    return pd.read_csv("data/population.csv")


def population_pipeline(df):
    # Input: the dataframe from load_input()

    # Sort by year and country, ascending
    df_sort = df.sort_values(by=["Entity", "Year"])
    # run a groupby to group by countries
    grouping = df_sort.groupby("Entity")
    first_row = grouping.first()
    last_row = grouping.last()
    # diff the year span and population
    delta_year = last_row["Year"] - first_row["Year"]
    delta_pop = (
        last_row["Population (historical)"] - first_row["Population (historical)"]
    )

    # find which indexes have more than 1 values and only keep those
    year_ranges = delta_year[delta_year > 0].index
    delta_pop = delta_pop.loc[year_ranges]
    delta_year = delta_year.loc[year_ranges]

    # divide the population diff by the length of the time period
    avg_year_delta = delta_pop / delta_year

    # get description
    avg_year_delta_desc = avg_year_delta.describe()

    # Return a list of min, median, max, mean, and standard deviation

    return [
        float(avg_year_delta_desc["min"]),
        float(avg_year_delta_desc["50%"]),
        float(avg_year_delta_desc["max"]),
        float(avg_year_delta_desc["mean"]),
        float(avg_year_delta_desc["std"]),
    ]


def q6():
    # As your answer to this part,
    # call load_input() and then population_pipeline()
    # Return a list of min, median, max, mean, and standard deviation
    return population_pipeline(load_input("data/population.csv"))


"""
7. Varying the input size

Next we want to set up three different datasets of different sizes.

Create three new files,
    - data/population-small.csv
      with the first 600 rows
    - data/population-medium.csv
      with the first 6000 rows
    - data/population-single-row.csv
      with only the first row
      (for calculating latency)

You can edit the csv file directly to extract the first rows
(remember to also include the header row)
and save a new file.

Make four versions of load input that load your datasets.
(The _large one should use the full population dataset.)
Each should return a dataframe.

The input CSV file will have 600 rows, but the DataFrame (after your cleaning) may have less than that.
"""


def load_input_small():
    return pd.read_csv("data/population-small.csv")


def load_input_medium():
    return pd.read_csv("data/population-medium.csv")


def load_input_large():
    return pd.read_csv("data/population.csv")


def load_input_single_row():
    # This is the pipeline we will use for latency.
    return pd.read_csv("data/population-single-row.csv")


def q7():
    # Don't modify this part
    s = load_input_small()
    m = load_input_medium()
    l = load_input_large()
    x = load_input_single_row()
    return [len(s), len(m), len(l), len(x)]


"""
8.
Create baseline pipelines

First let's create our baseline pipelines.
Create four pipelines,
    baseline_small
    baseline_medium
    baseline_large
    baseline_latency

based on the three datasets above.
Each should call your population_pipeline from Q6.

Your baseline_latency function will not be very interesting
as the pipeline does not produce any meaningful output on a single row!
You may choose to instead run an example with two rows,
or you may fill in this function in any other way that you choose
that you think is meaningful.
"""


def baseline_small():
    return population_pipeline(load_input_small())


def baseline_medium():
    return population_pipeline(load_input_medium())


def baseline_large():
    return population_pipeline(load_input_large())


def baseline_latency():
    # run an example with two rows.
    two_row = load_input_small().iloc[0:2, :]
    assert len(two_row) == 2
    return population_pipeline(two_row)


def q8():
    # Don't modify this part
    _ = baseline_medium()
    return ["baseline_small", "baseline_medium", "baseline_large", "baseline_latency"]


"""
9.
Finally, let's compare whether loading an input from file is faster or slower
than getting it from an existing Pandas dataframe variable.

Create four new dataframes (constant global variables)
directly in the script.
Then use these to write 3 new pipelines:
    fromvar_small
    fromvar_medium
    fromvar_large
    fromvar_latency

These pipelines should produce the same answers as in Q8.

As your answer to this part;
a. Generate a plot in output/part2-q9a.png of the throughputs
    Return the list of 6 throughputs in this order:
    baseline_small, baseline_medium, baseline_large, fromvar_small, fromvar_medium, fromvar_large
b. Generate a plot in output/part2-q9b.png of the latencies
    Return the list of 2 latencies in this order:
    baseline_latency, fromvar_latency
"""

# TODO
POPULATION_SMALL = pd.read_csv("data/population-small.csv")
POPULATION_MEDIUM = pd.read_csv("data/population-medium.csv")
POPULATION_LARGE = pd.read_csv("data/population.csv")
POPULATION_SINGLE_ROW = pd.read_csv("data/population-single-row.csv")
POPULATION_TWO_ROW = pd.read_csv("data/population-small.csv").iloc[
    0:2, :
]  # TODO: ask on piazza


def fromvar_small():
    return population_pipeline(POPULATION_SMALL)


def fromvar_medium():
    return population_pipeline(POPULATION_MEDIUM)


def fromvar_large():
    return population_pipeline(POPULATION_LARGE)


def fromvar_latency():
    # run an example with two rows.
    return population_pipeline(POPULATION_TWO_ROW)


def q9a():
    # Add all 6 pipelines for a throughput comparison
    h = ThroughputHelper()
    h.add_pipeline("baseline_small", len(POPULATION_SMALL), baseline_small)
    h.add_pipeline("baseline_medium", len(POPULATION_MEDIUM), baseline_medium)
    h.add_pipeline("baseline_large", len(POPULATION_LARGE), baseline_large)
    h.add_pipeline("fromvar_small", len(POPULATION_SMALL), fromvar_small)
    h.add_pipeline("fromvar_medium", len(POPULATION_MEDIUM), fromvar_medium)
    h.add_pipeline("fromvar_large", len(POPULATION_LARGE), fromvar_large)

    h.compare_throughput()
    h.generate_plot("output/part2-q9a.png")
    return h.throughputs


def q9b():
    h = LatencyHelper()
    h.add_pipeline("baseline_latency", baseline_latency)
    h.add_pipeline("fromvar_latency", fromvar_latency)
    h.compare_latency()
    h.generate_plot("output/part2-q9b.png")
    return h.latencies


"""
10.
Comment on the plots above!
How dramatic is the difference between the two pipelines?
Which differs more, throughput or latency?
What does this experiment show?

===== ANSWER Q10 BELOW =====
The difference in throughput is enormous on medium/large datasets, with the largest difference being 900,000 values/sec, but more minimal for smaller datasets.
The difference in latency is minimal, about 1.5*1e-6.
Throughput differs much more.
This experiment shows that loading to memory benefits large operations, which is expected, since RAM is very performant at extended random seek operations.
Latency isn't affected much, since there is only 1 random seek. The SSD is fast enough that loading 2 rows is going to be comparable to loading from RAM.
===== END OF Q10 ANSWER =====
"""

"""
===== Questions 11-14: Performance Comparison 2 =====

Our second performance comparison will explore vectorization.

Operations in Pandas use Numpy arrays and vectorization to enable
fast operations.
In particular, they are often much faster than using for loops.

Let's explore whether this is true!

11.
First, we need to set up our pipelines for comparison as before.

We already have the baseline pipelines from Q8,
so let's just set up a comparison pipeline
which uses a for loop to calculate the same statistics.

Your pipeline should produce the same answers as in Q6 and Q8.

Create a new pipeline:
- Iterate through the dataframe entries. You can assume they are sorted.
- Manually compute the minimum and maximum year for each country.
- Compute the same answers as in Q6.
- Manually compute the summary statistics for the resulting list (min, median, max, mean, and standard deviation).
"""


# TODO: Figure out why this is running like shit
def for_loop_pipeline(df):
    # Input: the dataframe from load_input()
    # Return a list of min, median, max, mean, and standard deviation

    # We need to iterate over each "entity" chunk until we hit the end of a block.
    avg_year_delta = []

    # Set up for loop with initial data (i.e. first row)
    current_row = df.iloc[0, :]
    current_entity = current_row["Entity"]  # get first row
    first_year = current_row["Year"]
    last_year = current_row["Year"]
    first_pop = current_row["Population (historical)"]
    last_pop = current_row["Population (historical)"]

    for i in range(1, len(df)):
        # Sample the ith row
        inspect_row = df.iloc[i, :]
        # If we are still in the chunk, and the new year is greater than the previous, increment years and population data.
        if inspect_row["Entity"] == current_entity:
            current_year = inspect_row["Year"]
            if current_year > first_year:
                last_year = current_year
                last_pop = inspect_row["Population (historical)"]
        else:
            # we're on a new country now, so calc averages
            if first_year != last_year:  # throw out single year examples
                avg_year_delta.append((last_pop - first_pop) / (last_year - first_year))
            # begin tracking the new country
            current_row = df.iloc[i, :]
            current_entity = current_row["Entity"]  # get first row
            first_year = current_row["Year"]
            last_year = current_row["Year"]
            first_pop = current_row["Population (historical)"]
            last_pop = current_row["Population (historical)"]

    # calculate the average for the final year as well
    if first_year != last_year:  # throw out single year examples
        avg_year_delta.append((last_pop - first_pop) / (last_year - first_year))

    # Return a list of min, median, max, mean, and standard deviation
    retlist = []

    # need to sort for median and min/max
    avg_year_delta.sort()
    # calculate min
    retlist.append(avg_year_delta[0])

    # calculate median
    if len(avg_year_delta) % 2 != 0:
        retlist.append(avg_year_delta[len(avg_year_delta) // 2])
    else:
        # average the centre 2 elements.
        mid1 = avg_year_delta[len(avg_year_delta) // 2 - 1]
        mid2 = avg_year_delta[len(avg_year_delta) // 2]
        retlist.append((mid1 + mid2) / 2)

    # calculate max
    retlist.append(avg_year_delta[len(avg_year_delta) - 1])

    # calculate mean
    accumulator = 0
    for val in avg_year_delta:
        accumulator += val
    mean = accumulator / len(avg_year_delta)
    retlist.append(float(mean))

    # calculate stdev
    numerator = 0
    if len(avg_year_delta) > 1:
        for val in avg_year_delta:
            numerator += (val - mean) ** 2
        retlist.append(
            float((numerator / (len(avg_year_delta) - 1)) ** 0.5)
        )  # note: pandas uses sample standard deviation

    return [float(item) for item in retlist]


def q11():
    # As your answer to this part, call load_input() and then
    # for_loop_pipeline() to return the 5 numbers.
    # (these should match the numbers you got in Q6.)
    return for_loop_pipeline(load_input("./data/population.csv"))


"""
12.
Now, let's create our pipelines for comparison.

As before, write 4 pipelines based on the datasets from Q7.
"""


def for_loop_small():
    return for_loop_pipeline(load_input_small())


def for_loop_medium():
    return for_loop_pipeline(load_input_medium())


def for_loop_large():
    return for_loop_pipeline(load_input_large())


def for_loop_latency():
    # run an example with two rows.
    two_row = load_input_small().iloc[0:2, :]
    assert len(two_row) == 2
    return for_loop_pipeline(two_row)


def q12():
    # Don't modify this part
    _ = for_loop_medium()
    return ["for_loop_small", "for_loop_medium", "for_loop_large", "for_loop_latency"]


"""
13.
Finally, let's compare our two pipelines,
as we did in Q9.

a. Generate a plot in output/part2-q13a.png of the throughputs
    Return the list of 6 throughputs in this order:
    baseline_small, baseline_medium, baseline_large, for_loop_small, for_loop_medium, for_loop_large

b. Generate a plot in output/part2-q13b.png of the latencies
    Return the list of 2 latencies in this order:
    baseline_latency, for_loop_latency
"""


def q13a():
    # Add all 6 pipelines for a throughput comparison
    # Generate plot in ouptut/q13a.png
    # Return list of 6 throughputs
    h = ThroughputHelper()
    h.add_pipeline("baseline_small", len(POPULATION_SMALL), baseline_small)
    h.add_pipeline("baseline_medium", len(POPULATION_MEDIUM), baseline_medium)
    h.add_pipeline("baseline_large", len(POPULATION_LARGE), baseline_large)
    h.add_pipeline("for_loop_small", len(POPULATION_SMALL), for_loop_small)
    h.add_pipeline("for_loop_medium", len(POPULATION_MEDIUM), for_loop_medium)
    h.add_pipeline("for_loop_large", len(POPULATION_LARGE), for_loop_large)

    h.compare_throughput()
    h.generate_plot("output/part2-q13a.png")
    return h.throughputs


def q13b():
    # Add 2 pipelines for a latency comparison
    # Generate plot in ouptut/q13b.png
    # Return list of 2 latencies
    h = LatencyHelper()
    h.add_pipeline("baseline_latency", baseline_latency)
    h.add_pipeline("for_loop_latency", for_loop_latency)
    h.compare_latency()
    h.generate_plot("output/part2-q13b.png")
    return h.latencies


"""
14.
Comment on the results you got!

14a. Which pipelines is faster in terms of throughput?

===== ANSWER Q14a BELOW =====
Vectorized pipeline is much faster in terms of throughput
===== END OF Q14a ANSWER =====

14b. Which pipeline is faster in terms of latency?

===== ANSWER Q14b BELOW =====
For Loop pipeline is faster in terms of latency
===== END OF Q14b ANSWER =====

14c. Do you notice any other interesting observations?
What does this experiment show?

===== ANSWER Q14c BELOW =====
Even though the for loop latency is mildly faster (~4ms), the throughput is completely awful, being several times slower than vectorized.
This experiment shows that, even though there is overhead from using pandas, it is likely not worth the effort to use for loops.
===== END OF Q14c ANSWER =====
"""

"""
===== Questions 15-17: Reflection Questions =====
15.

Take a look at all your pipelines above.
Which factor that we tested (file vs. variable, vectorized vs. for loop)
had the biggest impact on performance?

===== ANSWER Q15 BELOW =====
The factor of vectorized vs. for loop had the biggest impact on performance, especially in throughput (vectorization increases throughput several times over). However, the impact of file vs. variable
on throughput is still quite notable.
===== END OF Q15 ANSWER =====

16.
Based on all of your plots, form a hypothesis as to how throughput
varies with the size of the input dataset.

(Any hypothesis is OK as long as it is supported by your data!
This is an open ended question.)

===== ANSWER Q16 BELOW =====
Given the results in q13a.png and q9a.png, throughput is positively correlated with the size of the input dataset (at least in this synthetic test).
Regardless of vectorization/for loop or file/variable, the throughput increased with larger datasets.
===== END OF Q16 ANSWER =====

17.
Based on all of your plots, form a hypothesis as to how
throughput is related to latency.

(Any hypothesis is OK as long as it is supported by your data!
This is an open ended question.)

===== ANSWER Q17 BELOW =====
I would argue that throughput is negatively correlated to latency, even though this is not an apples to apples as latency operates on singular rows only.
Changes made to increase throughput (such as switching to vectorization or switching to in-RAM variables) either has no effect on latency, or causes latency to increase.
===== END OF Q17 ANSWER =====
"""

"""
===== Extra Credit =====

This part is optional.

Use your pipeline to compare something else!

Here are some ideas for what to try:
- the cost of random sampling vs. the cost of getting rows from the
  DataFrame manually
- the cost of cloning a DataFrame
- the cost of sorting a DataFrame prior to doing a computation
- the cost of using different encodings (like one-hot encoding)
  and encodings for null values
- the cost of querying via Pandas methods vs querying via SQL
  For this part: you would want to use something like
  pandasql that can run SQL queries on Pandas data frames. See:
  https://stackoverflow.com/a/45866311/2038713

As your answer to this part,
as before, return
a. the list of 6 throughputs
and
b. the list of 2 latencies.

and generate plots for each of these in the following files:
    output/part2-ec-a.png
    output/part2-ec-b.png
"""


# Extra credit (optional)
# Let's test the throughput of cloning dataframes and saving vs copying a source file directly.
# reuse the previous data.
#
def ec_helper_clone_df(data):
    newdf = data.copy()
    newdf.to_csv("data/ec/junk.csv")


def ec_helper_clone_df_small():
    ec_helper_clone_df(POPULATION_SMALL)


def ec_helper_clone_df_medium():
    ec_helper_clone_df(POPULATION_MEDIUM)


def ec_helper_clone_df_large():
    ec_helper_clone_df(POPULATION_LARGE)


def ec_helper_clone_df_single():
    ec_helper_clone_df(POPULATION_SINGLE_ROW)


def ec_helper_fs_copy(fname):
    with open("data/ec/junk.csv", "w") as f1:
        with open(fname) as f2:
            f2text = f2.read()
            f1.write(f2text)


def ec_helper_fs_copy_small():
    ec_helper_fs_copy("data/population-small.csv")


def ec_helper_fs_copy_medium():
    ec_helper_fs_copy("data/population-medium.csv")


def ec_helper_fs_copy_large():
    ec_helper_fs_copy("data/population.csv")


def ec_helper_fs_copy_single():
    ec_helper_fs_copy("data/population-single-row.csv")


def extra_credit_a():
    h = ThroughputHelper()
    h.add_pipeline(
        "ec_helper_fs_copy_small", len(POPULATION_SMALL), ec_helper_fs_copy_small
    )
    h.add_pipeline(
        "ec_helper_fs_copy_medium", len(POPULATION_MEDIUM), ec_helper_fs_copy_medium
    )
    h.add_pipeline(
        "ec_helper_fs_copy_large", len(POPULATION_LARGE), ec_helper_fs_copy_large
    )
    h.add_pipeline(
        "ec_helper_clone_df_small", len(POPULATION_SMALL), ec_helper_clone_df_small
    )
    h.add_pipeline(
        "ec_helper_clone_df_medium", len(POPULATION_MEDIUM), ec_helper_clone_df_medium
    )
    h.add_pipeline(
        "ec_helper_clone_df_large", len(POPULATION_LARGE), ec_helper_clone_df_large
    )
    h.compare_throughput()
    h.generate_plot("output/part2-ec-a.png")
    return h.throughputs


def extra_credit_b():
    h = LatencyHelper()
    h.add_pipeline("ec_helper_fs_copy_single", ec_helper_fs_copy_single)
    h.add_pipeline("ec_helper_clone_df_single", ec_helper_clone_df_single)
    h.compare_latency()
    h.generate_plot("output/part2-ec-b.png")
    return h.latencies


# python filesystem copy is much faster in both throughput and latency.

"""
===== Wrapping things up =====

**Don't modify this part.**

To wrap things up, we have collected
your answers and saved them to a file below.
This will be run when you run the code.
"""

ANSWER_FILE = "output/part2-answers.txt"
UNFINISHED = 0


def log_answer(name, func, *args):
    try:
        answer = func(*args)
        print(f"{name} answer: {answer}")
        with open(ANSWER_FILE, "a") as f:
            f.write(f"{name},{answer}\n")
            print(f"Answer saved to {ANSWER_FILE}")
    except NotImplementedError:
        print(f"Warning: {name} not implemented.")
        with open(ANSWER_FILE, "a") as f:
            f.write(f"{name},Not Implemented\n")
        global UNFINISHED
        UNFINISHED += 1


def PART_2_PIPELINE():
    open(ANSWER_FILE, "w").close()

    # Q1-5
    log_answer("q1", q1)
    log_answer("q2a", q2a)
    # 2b: commentary
    log_answer("q3", q3)
    log_answer("q4a", q4a)
    # 4b: commentary
    log_answer("q5a", q5a)
    log_answer("q5b", q5b)

    # Q6-10
    log_answer("q6", q6)
    log_answer("q7", q7)
    log_answer("q8", q8)
    log_answer("q9a", q9a)
    log_answer("q9b", q9b)
    # 10: commentary

    # Q11-14
    log_answer("q11", q11)
    log_answer("q12", q12)
    log_answer("q13a", q13a)
    log_answer("q13b", q13b)
    # 14: commentary

    # 15-17: reflection
    # 15: commentary
    # 16: commentary
    # 17: commentary

    # Extra credit
    log_answer("extra credit (a)", extra_credit_a)
    log_answer("extra credit (b)", extra_credit_b)

    # Answer: return the number of questions that are not implemented
    if UNFINISHED > 0:
        print("Warning: there are unfinished questions.")

    return UNFINISHED


"""
=== END OF PART 2 ===

Main function
"""

if __name__ == "__main__":
    log_answer("PART 2", PART_2_PIPELINE)
