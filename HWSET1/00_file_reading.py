'''
Lesson on file reading using Airline Safety Data
https://github.com/fivethirtyeight/data/tree/master/airline-safety
'''

# read the whole file at once, return a single string (including newlines)
# 'rU' mode (read universal) converts different line endings into '\n'
with open('airline-safety.csv', 'rU') as f:
    lines = f.read()

# use a context manager to automatically close your file


# read the file into a list (each list element is one row)
with open('airline-safety.csv', 'rU') as f:
    lines = [l.strip() for l in f]


# do the same thing using a list comprehension


# side note: splitting strings


# split each string (at the commas) into a list
split = [l.split(',') for l in lines]


# do the same thing using the csv module
import csv
with open('airline-safety.csv', 'rU') as f:
    reader = csv.reader(f, delimiter=',')
    split = [r for r in reader]


# separate the header and data
header = split[0]
data = split[1:]

'''
EXERCISES:
1. Create a list containing the average number of incidents per year for each airline.
Example for Aer Lingus: (2 + 0)/30 = 0.07
Expected output: [0.07, 2.73, 0.23, ...]
'''

# Your code goes here
[sum([float(a) for a in [r[2], r[5]]]) / 30 for r in data]


'''
2. Create a list of airline names (without the star).
Expected output: ['Aer Lingus', 'Aeroflot', 'Aerolineas Argentinas', ...]
'''

# Your code goes here
[r[0].replace('*', '') for r in data]


'''
3. Create a list (of the same length) that contains 1 if there's a star and 0 if not.
Expected output: [0, 1, 0, ...]
'''
# Your code goes here
[int('*' in r[0]) for r in data]


'''
4. BONUS: Create a dictionary in which the key is the airline name (without the star)
   and the value is the average number of incidents.
Expected output: {'Aer Lingus': 0.07, 'Aeroflot': 2.73, ...}
'''

# Your code goes here
avg_inc = [sum([float(a) for a in [r[2], r[5]]]) / 30 for r in data]
no_st = [r[0].replace('*', '') for r in data]
d = {a: b for (b, a) in zip(avg_inc, no_st)}


'''
A few extra things that will help you with the homework
'''

# 'set' data structure is useful for gathering unique elements
my_list = [1, 2, 1]
set(my_list)            # returns a set of 1, 2
len(set(my_list))       # count of unique elements

# 'in' statement is useful for lists
1 in my_list            # True
3 in my_list            # False

# 'in' is useful for strings (checks for substrings)
my_string = 'hello there'
'the' in my_string      # True
'then' in my_string     # False

# 'in' is useful for dictionaries (checks keys but not values)
my_dict = {'name':'Kevin', 'title':'instructor'}
'name' in my_dict       # True
'Kevin' in my_dict      # False

# 'count' method for strings counts how many times a character appears
my_string.count('e')    # 3
