"""
Part 1: Data Processing in Pandas

=== Instructions ===

There are 22 questions in this part.
For each part you will implement a function (q1, q2, etc.)
Each function will take as input a data frame
or a list of data frames and return the answer
to the given question.

To run your code, you can run `python3 part1.py`.
This will run all the questions that you have implemented so far.
It will also save the answers to part1-answers.txt.

=== Dataset ===

In this part, we will use a dataset of world university rankings
called the "QS University Rankings".

The ranking data was taken 2019--2021 from the following website:
https://www.topuniversities.com/university-rankings/world-university-rankings/2021

=== Grading notes ===

- Once you have completed this part, make sure that
  your code runs, that part1-answers.txt is being re-generated
  every time the code is run, and that the answers look
  correct to you.

- Be careful about output types. For example if the question asks
  for a list of DataFrames, don't return a numpy array or a single
  DataFrame. When in doubt, ask on Piazza!

- Make sure that you remove any NotImplementedError exceptions;
  you won't get credit for any part that raises this exception
  (but you will still get credit for future parts that do not raise it
  if they don't depend on the previous parts).

- Make sure that you fill in answers for the parts
  marked "ANSWER ___ BELOW" and that you don't modify
  the lines above and below the answer space.

- Q6 has a few unit tests to help you check your work.
  Make sure that you removed the `@pytest.mark.skip` decorators
  and that all tests pass (show up in green, no red text!)
  when you run `pytest part3.py`.

- For plots: There are no specific requirements on which
  plotting methods you use; if not specified, use whichever
  plot you think might be most appropriate for the data
  at hand.
  Please ensure your plots are labeled and human-readable.
  For example, call .legend() on the plot before saving it!

===== Questions 1-6: Getting Started =====

To begin, let's load the Pandas library.
"""

from math import e
import pandas as pd

"""
1. Load the dataset into Pandas

Our first step is to load the data into a Pandas DataFrame.
We will also change the column names
to lowercase and reorder to get only the columns we are interested in.

Implement the rest of the function load_input()
by filling in the parts marked TODO below.

Return as your answer to q1 the number of dataframes loaded.
(This part is implemented for you.)
"""

NEW_COLUMNS = [
    "rank",
    "university",
    "region",
    "academic reputation",
    "employer reputation",
    "faculty student",
    "citations per faculty",
    "overall score",
]


def load_input():
    """
    Input: None
    Return: a list of 3 dataframes, one for each year.
    """

    # Load the input files and return them as a list of 3 dataframes.
    df_2019 = pd.read_csv("data/2019.csv", encoding="latin-1")
    df_2020 = pd.read_csv("data/2020.csv", encoding="latin-1")
    df_2021 = pd.read_csv("data/2021.csv", encoding="latin-1")

    # Standardizing the column names
    df_2019.columns = df_2019.columns.str.lower()
    df_2020.columns = df_2019.columns.str.lower()
    df_2021.columns = df_2019.columns.str.lower()

    # Restructuring the column indexes
    # Fill out this part. You can use column access to get only the
    # columns we are interested in using the NEW_COLUMNS variable above.
    # Make sure you return the columns in the new order.

    # Note to self: python's equivalent to dplyr::select is called filter, and the equivalent to dplyr::filter is query
    # Select all columns according to NEW_COLUMNS, and assign their values to the df variables.
    df_2019 = df_2019.filter(items=NEW_COLUMNS)
    df_2020 = df_2020.filter(items=NEW_COLUMNS)
    df_2021 = df_2021.filter(items=NEW_COLUMNS)

    # When you are done, remove the next line...
    # raise NotImplementedError

    # ...and keep this line to return the dataframes.
    return [df_2019, df_2020, df_2021]


def q1(dfs):
    # As the "answer" to this part, let's just return the number of dataframes.
    # Check that your answer shows up in part1-answers.txt.
    return len(dfs)


"""
2. Input validation

Let's do some basic sanity checks on the data for Q1.

Check that all three data frames have the same shape,
and the correct number of rows and columns in the correct order.

As your answer to q2, return True if all validation checks pass,
and False otherwise.
"""


def q2(dfs):
    """
    Input: Assume the input is provided by load_input()

    Return: True if all validation checks pass, False otherwise.

    Make sure you return a Boolean!
    From this part onward, we will not provide the return
    statement for you.
    You can check that the "answers" to each part look
    correct by inspecting the file part1-answers.txt.
    """
    # Check:
    # - that all three dataframes have the same shape
    # - the number of rows
    # - the number of columns
    # - the columns are listed in the correct order

    # iterate over all dfs
    for df in dfs:
        # ensure that dfs is not empty:
        if len(dfs) == 0:
            return False

        # ensure that all have the same number of rows
        # this can be done by making sure that all match the first df
        if len(df) != len(dfs[0]):
            return False

        # ensure that there are the same number of columns as NEW_COLUMNS (also checks if column count is correct)
        if len(df.columns) != len(NEW_COLUMNS):
            return False

        # make sure all the columns listed are in the correct order
        for i in range(0, len(NEW_COLUMNS)):
            if df.columns[i] != NEW_COLUMNS[i]:
                return False

    return True


"""
===== Interlude: Checking your output so far =====

Run your code with `python3 part1.py` and open up the file
output/part1-answers.txt to see if the output looks correct so far!

You should check your answers in this file after each part.

You are welcome to also print out stuff to the console
in each question if you find it helpful.
"""

ANSWER_FILE = "output/part1-answers.txt"


def interlude():
    print("Answers so far:")
    with open(f"{ANSWER_FILE}") as fh:
        print(fh.read())


"""
===== End of interlude =====

3a. Input validation, continued

Now write a validate another property: that the set of university names
in each year is the same.
As in part 2, return a Boolean value.
(True if they are the same, and False otherwise)

Once you implement and run your code,
remember to check the output in part1-answers.txt.
(True if the checks pass, and False otherwise)
"""


def q3(dfs):
    # Check:
    # - that the set of university names in each year is the same
    # Return:
    # - True if they are the same, and False otherwise.

    # Check that all dfs match the first df's university set...
    university_0_set = set(dfs[0]["university"])

    # ...by iterating all university names as a set, and comparing against the first df.
    for df in dfs:
        if set(df["university"]) != university_0_set:
            return False

    return True


"""
3b (commentary).
Did the checks pass or fail?
Comment below and explain why.

=== ANSWER Q3b BELOW ===
Only the check for the dimensions (q2) passed, which means that not every year had the same universities.
This is expected since the datasets are CSVs of the top 100 universities for three different years; not every year will have the same universities at the same rankings.
A university in the top 100 for a certain year may be ranked outside of the top 100 for another year. This means that the university may not appear in all years (and another university will take its spot in the top 100s).
=== END OF Q3b ANSWER ===
"""

"""
4. Random sampling

Now that we have the input data validated, let's get a feel for
the dataset we are working with by taking a random sample of 5 rows
at a time.

Implement q4() below to sample 5 points from each year's data.

As your answer to this part, return the *university name*
of the 5 samples *for 2021 only* as a list.
(That is: ["University 1", "University 2", "University 3", "University 4", "University 5"])

Code design: You can use a for for loop to iterate over the dataframes.
If df is a DataFrame, df.sample(5) returns a random sample of 5 rows.

Hint:
    to get the university name:
    try .iloc on the sample, then ["university"].
"""

# array containing len(dfs) number of length 5 arrays
# len(dfs) x 5
df_samples = []


def q4(dfs):
    # Sample 5 rows from each dataframe
    # Print out the samples
    for df in dfs:
        sample = df.sample(5)
        print(sample)
        df_samples.append(sample)

    # Answer as a list of 5 university names

    solution = df_samples[2].iloc[:, [1]]["university"].to_list()
    return solution


"""
Once you have implemented this part,
you can run a few times to see different samples.

4b (commentary).
Based on the data, write down at least 2 strengths
and 3 weaknesses of this dataset.

=== ANSWER Q4b BELOW ===
Strengths:
1. The data is tidy: each variable is a column, each column is a variable; each observeration is a row, each row is an observation; each value is a cell, each cell is a value.
2. All datasets are of the same dimensions (with the same names), so operations between them are relatively simple (less/no transformation).

Weaknesses:
1. The datatypes are inconsistent. some values use integers (e.g. 100) and others appears to use floats (e.g. 99.8). They should be consistently using float values.
2. The dataset's naming and datatypes are inconsistent. We had to rename all the column names to lower case.
3. It is very cumbersome to have to load each csv individually. It would make sense to create a collated version of all years, with a column specifying a year for every position (e.g. using our current data, there would be 3 first places, with year values of 2019, 2020, and 2021).
=== END OF Q4b ANSWER ===
"""

"""
5. Data cleaning

Let's see where we stand in terms of null values.
We can do this in two different ways.

a. Use .info() to see the number of non-null values in each column
displayed in the console.

b. Write a version using .count() to calculate the number of
non-null values in each column.

In both 5a and 5b: return as your answer
*for the 2021 data only*
as a list of the number of non-null values in each column.

Example: if there are 5 non-null values in the first column, 3 in the second, 4 in the third, and so on, you would return
    [5, 3, 4, ...]
"""


def q5a(dfs):
    # TODO
    # Remember to return the list here
    # (Since .info() does not return any values,
    # for this part, you will need to copy and paste
    # the output as a hardcoded list.)

    print(dfs[2].info())
    return [100, 100, 100, 100, 100, 100, 100, 100]


def q5b(dfs):
    # TODO

    # return an array containing the non-nulls for each column of df[2]
    return dfs[2].count().to_list()


"""
5c.
One other thing:
Also fill this in with how many non-null values are expected.
We will use this in the unit tests below.
"""


def q5c():
    num_non_null = 100
    return num_non_null


"""
===== Interlude again: Unit tests =====

Unit tests

Now that we have completed a few parts,
let's talk about unit tests.
We won't be using them for the entire assignment
(in fact we won't be using them after this),
but they can be a good way to ensure your work is correct
without having to manually inspect the output.

We need to import pytest first.
"""

import pytest

"""
The following are a few unit tests for Q1-5.

To run the unit tests,
first, remove (or comment out) the `@pytest.mark.skip` decorator
from each unit test (function beginning with `test_`).
Then, run `pytest part1.py` in the terminal.
"""


# @pytest.mark.skip
def test_q1():
    dfs = load_input()
    assert len(dfs) == 3
    assert all([isinstance(df, pd.DataFrame) for df in dfs])


# @pytest.mark.skip
def test_q2():
    dfs = load_input()
    assert q2(dfs)


# @pytest.mark.skip
@pytest.mark.xfail
def test_q3():
    dfs = load_input()
    assert q3(dfs)


# @pytest.mark.skip
def test_q4():
    dfs = load_input()
    samples = q4(dfs)
    assert len(samples) == 5


# @pytest.mark.skip
def test_q5():
    dfs = load_input()
    answers = q5a(dfs) + q5b(dfs)
    assert len(answers) > 0
    num_non_null = q5c()
    for x in answers:
        assert x == num_non_null


"""
6a. Are there any tests which fail?

=== ANSWER Q6a BELOW ===
Test Q3 failed, since the top 100 universities are not identical every year.
=== END OF Q6a ANSWER ===

6b. For each test that fails, is it because your code
is wrong or because the test is wrong?

=== ANSWER Q6b BELOW ===
The test is wrong. My code correctly identifies that not every year will have the same top 100 universities. Instead, the test
should check the opposite case that every year has the same top 100 universities, which would be highly unlikely and would probably only happen if there was a code error.
=== END OF Q6b ANSWER ===

IMPORTANT: for any failing tests, if you think you have
not made any mistakes, mark it with
@pytest.mark.xfail
above the test to indicate that the test is expected to fail.
Run pytest part1.py again to see the new output.

6c. Return the number of tests that failed, even after your
code was fixed as the answer to this part.
(As an integer)
Please include expected failures (@pytest.mark.xfail).
(If you have no failing or xfail tests, return 0.)
"""


def q6c():
    # TODO
    return 1


"""
===== End of interlude =====

===== Questions 7-10: Data Processing =====

7. Adding columns

Notice that there is no 'year' column in any of the dataframe. As your first task, append an appropriate 'year' column in each dataframe.

Append a column 'year' in each dataframe. It must correspond to the year for which the data is represented.

As your answer to this part, return the number of columns in each dataframe after the addition.
"""


def q7(dfs):
    # TODO

    year_counter = 2019  # we can safely assume that the first df will always be 2019
    col_count = []  # list containing columns in each df
    # Iterate over all dfs and add the appropriate year per df
    for df in dfs:
        df["year"] = year_counter
        year_counter += 1
        col_count.append(len(df.columns))
    # Remember to return the list here
    return col_count


"""
8a.
Next, find the count of universities in each region that made it to the Top 100 each year. Print all of them.

As your answer, return the count for "USA" in 2021.
"""


def q8a(dfs):
    # Enter Code here

    university_counts = []  # list containing region stats for every df
    for df in dfs:
        # get regions in a df
        regions = df["region"].unique()

        region_stat = {}  # dictionary containing a region's stats, in the form ["region"] = count of universities
        for region in regions:
            region_stat[region] = len(df[df["region"] == region])
        university_counts.append(region_stat)

    # print the count of universities in each region for all 3 years.
    year = 2019
    for i in range(0, 3):
        print(f"Year: {year}")
        print(university_counts[i])
        year += 1

    # Remember to return the count here
    return university_counts[2]["USA"]


"""
8b.
Do you notice some trend? Comment on what you observe and why might that be consistent throughout the years.

=== ANSWER Q8b BELOW ===
Yes, the ordering of some countries from most to least schools in the top 100s remains relatively consistent throughout the years.
USA, UK, Switzerland, Singapore, China, Japan, Australia, Canada, and South Korea all keep the roughly the same order throughout 2019-2021, and even maintain relatively the same number of top 100 schools.
This is likely because those countries have well-established universities which are not likely to stop existing or deterioriate.
=== END OF Q8b ANSWER ===
"""

"""
9.
From the data of 2021, find the average score of all attributes for all universities.

As your answer, return the list of averages (for all attributes)
in the order they appear in the dataframe:
academic reputation, employer reputation, faculty student, citations per faculty, overall score.

The list should contain 5 elements.
"""


def q9(dfs):
    # Enter code here
    # TODO

    # There are non-numeric columns, so we cannot include them
    numeric_cols = [
        "academic reputation",
        "employer reputation",
        "faculty student",
        "citations per faculty",
        "overall score",
    ]

    # list to be returned
    ret_list = []

    # Average for all cols
    target_year = dfs[2]
    for col in target_year.columns:
        if col in numeric_cols:
            ret_list.append(target_year[col].mean().item())

    # Return the list here
    return ret_list


"""
10.
From the same data of 2021, now find the average of *each* region for **all** attributes **excluding** 'rank' and 'year'.

In the function q10_helper, store the results in a variable named **avg_2021**
and return it.

Then in q10, print the first 5 rows of the avg_2021 dataframe.
"""


def q10_helper(dfs):
    # Enter code here
    # TODO
    # Placeholder for the avg_2021 dataframe
    avg_2021 = pd.DataFrame()

    # desired year is 2021
    target_year = dfs[2]

    # define groups of regions
    regions = set(target_year["region"])

    # get a list of all the desired colnames
    col_names = [
        x for x in target_year if x != "rank" and x != "year" and x != "university"
    ]

    # define columns in avg_2021
    for col in col_names:
        avg_2021[col] = 0  # Initialize the column

    # iterate over regions
    for region in regions:
        # subset the dataframes to only contain the desired region
        temp_df = target_year[target_year["region"] == region]

        # define a dict representing a row in avg_2021
        row = {}
        for col in col_names:
            if col not in ("region"):
                row[col] = temp_df[col].mean().item()
            elif col in ("region"):
                row[col] = temp_df[col].values[0]

        avg_2021 = pd.concat([avg_2021, pd.DataFrame([row])])

    return avg_2021


def q10(avg_2021):
    """
    Input: the avg_2021 dataframe
    Print: the first 5 rows of the dataframe

    As your answer, simply return the number of rows printed.
    (That is, return the integer 5)
    """
    # Enter code here
    print(avg_2021.head(5))
    # Return 5
    return 5


"""
===== Questions 11-14: Exploring the avg_2021 dataframe =====

11.
Sort the avg_2021 dataframe from the previous question based on overall score in a descending fashion (top to bottom).

As your answer to this part, return the first row of the sorted dataframe.
"""


def q11(avg_2021):
    avg_2021 = avg_2021.sort_values(by="overall score", ascending=False)
    return avg_2021.take([0])


"""
12a.
What do you observe from the table above? Which country tops the ranking?

What is one country that went down in the rankings
between 2019 and 2021?

You will need to load the data from 2019 to get the answer to this part.
You may choose to do this
by writing another function like q10_helper and running q11,
or you may just do it separately
(e.g., in a Python shell) and return the name of the country/region
that you found went down in the rankings.

Errata: please note that the 2021 dataset we provided is flawed
(it is almost identical to the 2020 data).
This is why the question now asks for the difference between 2019 and 2021.

For the answer to this part return the name of the country/region that tops the ranking
and the name of one country/region that went down in the rankings.
"""


# Note: I ran the dataset separately.
def q12a(avg_2021):
    return ("Singapore", "South Korea")


"""
12b.
Comment on why the country above is at the top of the list.
(Note: This is an open-ended question.)

=== ANSWER Q12b BELOW ===
Singapore is likely at the top of the list as it is an extremely small, but wealthy country.
This would likely mean that the students attending will likely have had more educational opportunity and focus from the government, allowing for better performance in university.
=== END OF Q12b ANSWER ===
"""

"""
13a.
Represent all the attributes in the avg_2021 dataframe using a box and whisker plot.

Store your plot in output/part1-13a.png.

As the answer to this part, return the name of the plot you saved.

**Hint:** You can do this using subplots (and also otherwise)
"""

import matplotlib.pyplot as plt


def q13a(avg_2021):
    # Plot the box and whisker plot
    # TODO : Do we drop region?

    # Drop region
    avg_2021_no_region = avg_2021.drop(columns=["region"])

    # Create a 1d subplot
    fig, axes = plt.subplots(ncols=len(avg_2021_no_region.columns), nrows=1)
    fig.suptitle("Box Plots of avg_2021")

    # Iterate over all columns of region and plot in the subplots
    for i in range(len(avg_2021_no_region.columns)):
        axes[i].boxplot(x=avg_2021_no_region.iloc[:, i], vert=True)
        axes[i].set_title(avg_2021_no_region.columns[i], fontsize=8)
        axes[i].set_ylabel("Score")

    plt.tight_layout()
    plt.savefig("output/part1-13a.png")
    return "output/part1-13a.png"


"""
b. Do you observe any anomalies in the box and whisker
plot?

=== ANSWER Q13b BELOW ===
There is an outlier for the overall score plot. This is represented by the circle (one country has a significantly higher overall score).
This outlier is likely Singapore, which we identified as the highest scoring nation.
=== END OF Q13b ANSWER ===
"""

"""
14a.
Pick two attributes in the avg_2021 dataframe
and represent them using a scatter plot.

Store your plot in output/part1-14a.png.

As the answer to this part, return the name of the plot you saved.
"""


def q14a(avg_2021):
    # Enter code here
    # TODO

    plt.clf()  # otherwise the previous plot will shadow this

    # choose two attributes
    dual_plot = avg_2021.loc[:, ["academic reputation", "citations per faculty"]]

    # create scatterplot
    plt.scatter(
        dual_plot.loc[:, ["academic reputation"]],
        dual_plot.loc[:, ["citations per faculty"]],
    )

    plt.title("Citations per faculty vs. academic reputation")
    plt.tight_layout()
    plt.savefig("output/part1-14a.png")
    return "output/part1-14a.png"


"""
Do you observe any general trend?

=== ANSWER Q14b BELOW ===
There appears to be a very weak positive correlation between the two attributes.
=== END OF Q14b ANSWER ===

===== Questions 15-20: Exploring the data further =====

We're more than halfway through!

Let's come to the Top 10 Universities and observe how they performed over the years.

15. Create a smaller dataframe which has the top ten universities from each year, and only their overall scores across the three years.

Hint:

*   There will be four columns in the dataframe you make
*   The top ten universities are same across the three years. Only their rankings differ.
*   Use the merge function. You can read more about how to use it in the documentation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html
*   Shape of the resultant dataframe should be (10, 4)

As your answer, return the shape of the new dataframe.
"""


def q15_helper(dfs):
    # Return the new dataframe
    # TODO
    # Placeholder:

    # The columns are university name, 2019 score, 2020 score, 2021 score
    top_10 = pd.DataFrame()

    # year = 2019

    for i in range(len(dfs)):
        # Make sure the df is sorted by rank (just in case)
        current_year_df_sort = dfs[i].sort_values(by="rank", ascending=True)

        # Subset the top 10 schools
        current_year_df = current_year_df_sort.iloc[0:10, :]

        # only keep the university name and overall score
        current_year_df = current_year_df.loc[:, ["university", "overall score"]]

        # merge to top_10
        if i == 0:
            top_10 = current_year_df  # merge doesn't work on itself
        else:
            top_10 = top_10.merge(
                current_year_df,
                on="university",
                how="outer",
            )
    return top_10


def q15(top_10):
    # Enter code here
    # TODO
    return (len(top_10), len(top_10.columns))


"""
16.
You should have noticed that when you merged,
Pandas auto-assigned the column names. Let's change them.

For the columns representing scores, rename them such that they describe the data that the column holds.

You should be able to modify the column names directly in the dataframe.
As your answer, return the new column names as a list.
"""


def q16(top_10):
    # Enter code here
    # TODO
    top_10_clone = top_10

    # rename cols to describe overall score for each year
    year = 2019
    for i in range(1, 4):
        top_10_clone.rename(
            columns={top_10.columns[i]: f"overall score_{year}"}, inplace=True
        )
        year += 1
    return list(top_10_clone.columns)


"""
17a.
Draw a suitable plot to show how the overall scores of the Top 10 universities varied over the three years. Clearly label your graph and attach a legend. Explain why you chose the particular plot.

Save your plot in output/part1-17a.png.

As the answer to this part, return the name of the plot you saved.

Note:
*   All universities must be in the same plot.
*   Your graph should be clear and legend should be placed suitably
"""


def q17a(top_10):
    # I chose to use a line plot, since we want to show change in overall scores over a year range.
    # A line plot is suited for displaying the relation between the two variables (since it displays the relationship between the y and x variable),
    # and the line helps to show trends in the data.

    # Enter code here
    # TODO
    plt.clf()  # clear graphs

    # Set up titles
    plt.title("Top 10 Schools' Overall Score from 2019-2021")
    plt.xlabel("Year")
    plt.ylabel("Overall Score")
    plt.xticks(range(2019, 2022, 1))

    # iterate plot over the top 10 schools
    for i in range(len(top_10)):
        year_range = list(range(2019, 2022, 1))
        overall_scores = top_10.iloc[i, 1:4]

        uni_name = top_10.iloc[i, 0]

        plt.plot(year_range, overall_scores, label=uni_name)

    # show legend
    plt.legend(prop={"size": 6}, bbox_to_anchor=(1, 1))
    # plt.show()

    plt.tight_layout()
    plt.savefig("output/part1-17a.png")
    return "output/part1-17a.png"


"""
17b.
What do you observe from the plot above? Which university has remained consistent in their scores? Which have increased/decreased over the years?

=== ANSWER Q17b BELOW ===
While the data is technically flawed between 2020 and 2021, there are some noticeable trends.
MIT had no change in scoring.
Stanford, Harvard, CIT, Cambridge, and UChicago all had decreases over the years.
Oxford, Zurich, UCL, and Imperial all had increases over the years.
=== END OF Q17b ANSWER ===
"""

"""
===== Questions 18-19: Correlation matrices =====

We're almost done!

Let's look at another useful tool to get an idea about how different variables are corelated to each other. We call it a **correlation matrix**

A correlation matrix provides a correlation coefficient (a number between -1 and 1) that tells how strongly two variables are correlated. Values closer to -1 mean strong negative correlation whereas values closer to 1 mean strong positve correlation. Values closer to 0 show variables having no or little correlation.

You can learn more about correlation matrices from here: https://www.statology.org/how-to-read-a-correlation-matrix/

18.
Plot a correlation matrix to see how each variable is correlated to another. You can use the data from 2021.

Print your correlation matrix and save it in output/part1-18.png.

As the answer to this part, return the name of the plot you saved.

**Helpful link:** https://datatofish.com/correlation-matrix-pandas/
"""


def q18(dfs):
    # Enter code here
    # TODO
    plt.clf()

    df_2021 = dfs[2]

    # drop year since it is blank
    df_2021 = df_2021.drop(
        ["region", "university", "year"], axis=1
    )  # non-numeric / irrelevant data must be removed

    # required to add title / axis labels
    fig, ax = plt.subplots()

    plt.xticks(range(len(df_2021.columns)), df_2021, fontsize=6, rotation=25)
    plt.yticks(range(len(df_2021.columns)), df_2021, fontsize=8)
    fig.suptitle("Correlation Matrix for 2021 Top 100 Universities", y=0.05)

    ax.matshow(df_2021.corr(), cmap="viridis")
    plt.savefig("output/part1-18.png")
    plt.clf()  # clear graphs before part 2
    return "output/part1-18.png"


"""
19. Comment on at least one entry in the matrix you obtained in the previous
part that you found surprising or interesting.

=== ANSWER Q19 BELOW ===
I found it surprising that there is moderately strong positive correlation between faculty student and employer reputation (indicated by the color green, according to the viridis colorset).
I would have thought that employers were only interested in how many citations a university puts out, and less with the statistics relating to the inner workings of the university (like faculty-student ratio)
=== END OF Q19 ANSWER ===
"""

"""
===== Questions 20-23: Data manipulation and falsification =====

This is the last section.

20. Exploring data manipulation and falsification

For fun, this part will ask you to come up with a way to alter the
rankings such that your university of choice comes out in 1st place.

The data does not contain UC Davis, so let's pick a different university.
UC Berkeley is a public university nearby and in the same university system,
so let's pick that one.

We will write two functions.
a.
First, write a function that calculates a new column
(that is you should define and insert a new column to the dataframe whose value
depends on the other columns)
and calculates
it in such a way that Berkeley will come out on top in the 2021 rankings.

Note: you can "cheat"; it's OK if your scoring function is picked in some way
that obviously makes Berkeley come on top.
As an extra challenge to make it more interesting, you can try to come up with
a scoring function that is subtle!

b.
Use your new column to sort the data by the new values and return the top 10 university names as a list.

"""


def q20a(dfs):
    # TODO

    # We can make a fake column, "Upward Potential in California" that penalizes having a high ranking of employer reputation in California,
    # but benefits a high academic reputation

    df_2021 = dfs[2]

    local_focus = []
    for i in range(len(df_2021)):
        # figure out if the university is in california
        if df_2021.loc[i, "university"].find("California") == -1:
            cali = 0
        else:
            cali = 1

        # calculate a penalty score
        local_focus.append(
            min(
                100,
                (
                    100
                    - df_2021.loc[i, "employer reputation"]
                    + df_2021.loc[i, "academic reputation"] * 0.25
                )
                * cali,
            )
        )

    # add to the original df
    df_2021["upward potential"] = local_focus

    # For your answer, return the score for Berkeley in the new column.
    return (
        df_2021[df_2021["university"] == "University of California, Berkeley (UCB)"]
        .loc[:, "upward potential"]
        .values[0]
    )


def q20b(dfs):
    # TODO
    # For your answer, return the top 10 university names as a list.
    # sort
    df_2021 = dfs[2].sort_values("upward potential", ascending=False)

    # append to returnlist
    ret_list = []
    for i in range(10):
        ret_list.append(df_2021.iloc[i, 1])

    return ret_list


"""
21. Exploring data manipulation and falsification, continued

This time, let's manipulate the data by changing the source files
instead.
Create a copy of data/2021.csv and name it
data/2021_falsified.csv.
Modify the data in such a way that UC Berkeley comes out on top.

For this part, you will also need to load in the new data
as part of the function.
The function does not take an input; you should get it from the file.

Return the top 10 university names as a list from the falsified data.
"""


def q21():
    # for practice's sake, let's try copying the old data, modding it, and then saving it to duplicate

    # load old 2021 data
    df_2021 = pd.read_csv("data/2021.csv", encoding="latin-1")

    # set all fields to 100 if berkeley
    berk = df_2021[df_2021["University"] == "University of California, Berkeley (UCB)"]
    for i in range(len(berk.columns)):
        # but don't modify the text fields
        if i not in (0, 1, 9):
            berk.iloc[0, i] = 100
        if i == 0:
            berk.iloc[0, i] = 1  # uprank

    # save it back to 2021_falsified.csv
    df_2021.loc[df_2021["University"] == "University of California, Berkeley (UCB)"] = (
        berk
    )

    # downgrade MIT because they rank too high
    mit = df_2021[
        df_2021["University"] == "Massachusetts Institute of Technology (MIT)"
    ]
    for i in range(len(mit.columns)):
        if i not in (0, 1, 9):
            mit.iloc[0, i] = 99.9
        if i == 0:
            mit.iloc[0, i] = 28  # derank
    df_2021.loc[
        df_2021["University"] == "Massachusetts Institute of Technology (MIT)"
    ] = mit

    # save the fake data
    df_2021 = df_2021.sort_values("Rank", ascending=True)
    df_2021.to_csv("data/2021_falsified.csv", encoding="latin-1", index=False)

    # load the fake data
    fake_data = pd.read_csv("data/2021_falsified.csv", encoding="latin-1")
    fake_data = fake_data.sort_values("Overall Score", ascending=False)

    return list(fake_data.iloc[0:10, 1])
    # return list(fake_data.iloc[0:10, 1])

    # return as a list


"""
22. Exploring data manipulation and falsification, continued

Which of the methods above do you think would be the most effective
if you were a "bad actor" trying to manipulate the rankings?

Which do you think would be the most difficult to detect?

=== ANSWER Q22 BELOW ===
The first method of making a misleading statistic is far more effective and difficult to detect than simply overwriting data.
In the first method, people can be easily fooled into believing it is a legitimate number, and it is very easy to make a misleading justification for the score.
The second method is more obvious and difficult to justify.
=== END OF Q22 ANSWER ===
"""

"""
===== Wrapping things up =====

To wrap things up, we have collected
everything together in a pipeline for you
below.

**Don't modify this part.**
It will put everything together,
run your pipeline and save all of your answers.

This is run in the main function
and will be used in the first part of Part 2.
"""

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


def PART_1_PIPELINE():
    open(ANSWER_FILE, "w").close()

    try:
        dfs = load_input()
    except NotImplementedError:
        print("Welcome to Part 1! Implement load_input() to get started.")
        dfs = []

    # Questions 1-6
    log_answer("q1", q1, dfs)
    log_answer("q2", q2, dfs)
    log_answer("q3a", q3, dfs)
    # 3b: commentary
    log_answer("q4", q4, dfs)
    # 4b: commentary
    log_answer("q5a", q5a, dfs)
    log_answer("q5b", q5b, dfs)
    log_answer("q5c", q5c)
    # 6a: commentary
    # 6b: commentary
    log_answer("q6c", q6c)

    # Questions 7-10
    log_answer("q7", q7, dfs)
    log_answer("q8a", q8a, dfs)
    # 8b: commentary
    log_answer("q9", q9, dfs)
    # 10: avg_2021
    avg_2021 = q10_helper(dfs)
    log_answer("q10", q10, avg_2021)

    # Questions 11-15
    log_answer("q11", q11, avg_2021)
    log_answer("q12", q12a, avg_2021)
    # 12b: commentary
    log_answer("q13", q13a, avg_2021)
    # 13b: commentary
    log_answer("q14a", q14a, avg_2021)
    # 14b: commentary

    # Questions 15-17
    top_10 = q15_helper(dfs)
    log_answer("q15", q15, top_10)
    log_answer("q16", q16, top_10)
    log_answer("q17", q17a, top_10)
    # 17b: commentary

    # Questions 18-20
    log_answer("q18", q18, dfs)
    # 19: commentary

    # Questions 20-22
    log_answer("q20a", q20a, dfs)
    log_answer("q20b", q20b, dfs)
    log_answer("q21", q21)
    # 22: commentary

    # Answer: return the number of questions that are not implemented
    if UNFINISHED > 0:
        print("Warning: there are unfinished questions.")

    return UNFINISHED


"""
That's it for Part 1!

=== END OF PART 1 ===

Main function
"""

if __name__ == "__main__":
    log_answer("PART 1", PART_1_PIPELINE)
