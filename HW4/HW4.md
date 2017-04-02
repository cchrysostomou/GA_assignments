# HW SET 4: Command line tasks #
____

### Get the file ###
Rather than clone the whole repository, just pull down the chipotle.tsv file

**the http path will be the same as the url for viewing the RAW chipotle.tsv path on github site**

```
wget -O chipotle.tsv https://raw.githubusercontent.com/ga-students/DS-SEA-06/master/data/chipotle.tsv?token={githubusertokennamehere}
```
____

### Look at file structure ###

#### *Look at the head of the file* ####

```
head chipotle.tsv
```

The top 10 lines look like:

| order_id | quantity | item_name | choice_description | item_price |
| -------- | -------- | --------- | ------------------ | ---------- |
|1 |  1 |  Chips and Fresh Tomato Salsa  |  NULL  |  $2.39 |
|1 | 1  | Izze  |  [Clementine]   | $3.39 |
|1 |  1 |  Nantucket Nectar  |  [Apple] | $3.39 |
|1 |  1 |  Chips and Tomatillo-Green Chili Salsa |  NULL   | $2.39 |
|2 |  2 |  Chicken Bowl  |  [Tomatillo-Red Chili Salsa (Hot), [Black Beans, Rice, Cheese, Sour Cream]]  | $16.98 |
|3 |  1 |  Chicken Bowl  |  [Fresh Tomato Salsa (Mild), [Rice, Cheese, Sour Cream, Guacamole, Lettuce]] | $10.98 |
|3 |  1 |  Side of Chips |  NULL  |  $1.69 |
|4 |  1 |  Steak Burrito |  [Tomatillo Red Chili Salsa, [Fajita Vegetables, Black Beans, Pinto Beans, Cheese, Sour Cream, Guacamole, Lettuce]] |  $11.75 |
|4 | 1  | Steak Soft Tacos  |  [Tomatillo Green Chili Salsa, [Pinto Beans, Cheese, Sour Cream, Lettuce]] |  $9.25 |


#### *Look at the tail of the file* ####

```
tail chipotle.tsv
```

The last 10 lines look like:

| order_id | quantity | item_name | choice_description | item_price |
| -------- | -------- | --------- | ------------------ | ---------- |
1831 |   1  | Carnitas Bowl|[Fresh Tomato Salsa, [Fajita Vegetables, Rice, Black Beans, Cheese, Sour Cream, Lettuce]]|$9.25
1831| 1|Chips|NULL| $2.15
1831| 1|Bottled Water|NULL| $1.50
1832| 1|Chicken Soft Tacos | [Fresh Tomato Salsa, [Rice, Cheese, Sour Cream]]| $8.75
1832| 1|Chips and Guacamole | NULL| $4.45
1833| 1|Steak Burrito | [Fresh Tomato Salsa, [Rice, Black Beans, Sour Cream, Cheese, Lettuce, Guacamole]]|$11.75
1833| 1|Steak Burrito | [Fresh Tomato Salsa, [Rice, Sour Cream, Cheese, Lettuce, Guacamole]] | $11.75
1834| 1|Chicken Salad Bowl |  [Fresh Tomato Salsa, [Fajita Vegetables, Pinto Beans, Guacamole, Lettuce]] |  $11.25
1834| 1|Chicken Salad Bowl | [Fresh Tomato Salsa, [Fajita Vegetables, Lettuce]] | $8.75
1834| 1|Chicken Salad Bowl | [Fresh Tomato Salsa, [Fajita Vegetables, Pinto Beans, Lettuce]] | $8.75

#### *What does each row mean* ####

Each row represents a unique item that has been purchased *within* a chipotle order

#### *What each column means* ####

| Column | Description |
| ---  | --- |
| order_id | An identifier to map a specific row to an entire chipotle purchase |
| quantity | # of those items ordered (i.e. 5 Chicken Bowls) |
| item_name | Name of the item/row |
| choice_description | list of sub-items that make up/are within the the item (i.e. ingredients in item) |
| item_price | $ost of the item (quantity*unit price)|

### Number of items in table ###

#### *How many orders do there appear to be* ####

If the table was not sorted, this would not work...but, if it is sorted and increasing by 1, then the last line should have the last order # and therefore total orders

`tail -1 chipotle.tsv | cut -f1`

**Output:**

1834

#### *How many lines do there appear to be* ####

`wc -l chipotle.tsv | cut -d ' ' -f1`

**Output:**

4623

### General stats ###

#### *Which burrito is more popular, steak or chicken* ####

```
grep "Steak Burrito" chipotle.tsv | wc -l
grep "Chicken Burrito" chipotle.tsv | wc -l
```

**Output:**

368

553

Chicken is more popular

#### *Do chicken burritos more often have black beans or pinto beans?* ####

```
grep -i -E "Chicken Burrito.*black beans" chipotle.tsv | wc -l
grep -i -E "Chicken Burrito.*pinto beans" chipotle.tsv | wc -l
```

**Output:**

282

105

Black beans are more popular

### Github CSV files ###

#### *Make a list of all of the CSV or TSV files in [our class repo]* ####

Assuming we have pulled the github repo in some [folder_path]...

`find {folder_path} -iregex ".*.[c|t]sv" | rev | cut -d / -f1 | rev`

**Description: **
1. find all files within {folder_path}
 * Use iregex to search for a regular expression of all files ending in either *.csv or *.tsv
2. Return only the last column in the find result (i.e. remove the upstream path)
 * Using rev: Flip the string around
 * Using cut -d / -f1: Take the first column of the reversed string after seperated by /
 * Using rev: print the original direction of the reversed string (reversed it again)


**Output:**

NBA_players_2015.csv<br>
yelp.csv<br>
titanic.csv<br>
chipotle.tsv<br>
rt_critics.csv<br>
bank-additional.csv<br>
rossmann.csv<br>
hitters.csv<br>
drones.csv<br>
imdb_1000.csv<br>
airlines.csv<br>
vehicles_test.csv<br>
Airline_on_time_west_coast.csv<br>
syria.csv<br>
time_series_train.csv<br>
bikeshare.csv<br>
drinks.csv<br>
2015_trip_data.csv<br>
2015_weather_data.csv<br>
2015_station_data.csv<br>
vehicles_train.csv<br>
mtcars.csv<br>
citibike_feb2014.csv<br>
sms.tsv<br>
ozone.csv<br>
icecream.csv<br>
stores.csv<br>
ufo.csv<br>


#### *Count the approximate number of occurrences of the word "dictionary" (regardless of the case) across all files of [our class repo]* ####

`grep -iro "dictionary" {folder_path} | wc -l`

**Description: **
1. grep -iro: recursively grep within all files of the provided folder_path (r). print each occurrence rather than file/line containing string (o). Use case insensitive search (i)
2. Print the total number of lines reported

**Output:**

81

#### *Discover something "interesting" about the Chipotle data* ####
**Show the top ten most expensive orders**

```
awk '
    BEGIN{FS="\t"}
    NR>1{order[$1]+=substr($NF, 2)}
    END{for (i in order){print("Order", i, "=",  order[i])}}
' chipotle.tsv | sort -k4,4nr | head -10
```

**Output:**

Order 926 = 205.25<br>
Order 1443 = 160.74<br>
Order 1483 = 139<br>
Order 691 = 118.25<br>
Order 1786 = 114.3<br>
Order 205 = 109.9<br>
Order 511 = 104.59<br>
Order 491 = 102<br>
Order 1449 = 95.39<br>
Order 759 = 86.3<br>


**Show the last ten most expensive orders**

```
awk '
    BEGIN{FS="\t"}
    NR>1{order[$1]+=substr($NF, 2)}
    END{for (i in order){print("Order", i, "=",  order[i])}}
' chipotle.tsv | sort -k4,4nr | tail -10
```

**Output:**
Order 388 = 10.08<br>
Order 393 = 10.08<br>
Order 47 = 10.08<br>
Order 525 = 10.08<br>
Order 55 = 10.08<br>
Order 614 = 10.08<br>
Order 730 = 10.08<br>
Order 788 = 10.08<br>
Order 87 = 10.08<br>
Order 889 = 10.08<br>



**Description:**

This will calculate the total cost of each order

1. BEGIN{FS="\t"}: Tell AWK that the file is tab separated
2. NR>1{order[$1]+=substr($NF,2)}: Ignoring the first line (NR>1...all line numbers greater than one), create an associative array/key-value hash/dict where the key is the current order number. For each row update the array to store the cost of the item using the last column ($NF) in the line (substr($NF, 2)..use substring to remove the '$' character)
3. END{for (i in order){print("Order", i, "=",  order[i])}}: Once complete, print out the associative array (Order number, and total cost)
4. sort -k4,4nr: Numerically sort what was printed out by the last column so that we report costs from most expensive to least expensive


**The cost of orders ranged from $205.25 to $10.08, but the distribution of costs will not be not normal**
