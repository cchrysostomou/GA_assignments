#HW SET 4#
____

### Get the file ###
Rather than clone the whole repository, just pull down the chipotle.tsv file

**the http path will be the same as the url for viewing the RAW chipotle.tsv path on github site**

```
wget -O chipotle.tsv https://raw.githubusercontent.com/ga-students/DS-SEA-06/master/data/chipotle.tsv?token={githubusertokennamehere}
```
____

### Look at file structure ###

** Look at the head of the file **

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


** Look at the tail of the file **

```
tail chipotle.tsv
```

The last 10 lines look like:

1831    1   Carnitas Bowl   [Fresh Tomato Salsa, [Fajita Vegetables, Rice, Black Beans, Cheese, Sour Cream, Lettuce]]   $9.25
1831    1   Chips   NULL    $2.15
1831    1   Bottled Water   NULL    $1.50
1832    1   Chicken Soft Tacos  [Fresh Tomato Salsa, [Rice, Cheese, Sour Cream]]    $8.75
1832    1   Chips and Guacamole NULL    $4.45
1833    1   Steak Burrito   [Fresh Tomato Salsa, [Rice, Black Beans, Sour Cream, Cheese, Lettuce, Guacamole]]   $11.75
1833    1   Steak Burrito   [Fresh Tomato Salsa, [Rice, Sour Cream, Cheese, Lettuce, Guacamole]]    $11.75
1834    1   Chicken Salad Bowl  [Fresh Tomato Salsa, [Fajita Vegetables, Pinto Beans, Guacamole, Lettuce]]  $11.25
1834    1   Chicken Salad Bowl  [Fresh Tomato Salsa, [Fajita Vegetables, Lettuce]]  $8.75
1834    1   Chicken Salad Bowl  [Fresh Tomato Salsa, [Fajita Vegetables, Pinto Beans, Lettuce]] $8.75

** What does each row mean **

Each row represents a unique item that has been purchased *within* a chipotle order

** What each column means **

| Column | Description |
| ---  | --- |
| order_id | An identifier to map a specific row to an entire chipotle purchase |
| quantity | # of those items ordered (i.e. 5 Chicken Bowls) |
| item_name | Name of the item/row |
| choice_description | list of sub-items that make up/are within the the item (i.e. ingredients in item) |
| item_price | $ost or the individual item purchased |

### Number of items in table ###

** How many orders do there appear to be **

If the table was not sorted, this would not work...but, if it is sorted and increasing by 1, then the last line should have the last order # and therefore total orders

`tail -1 chipotle.tsv | cut -f1`

**Output: **1834

** How many lines do there appear to be **

`wc -l chipotle.tsv | cut -d ' ' -f1`

**Output: **4623

### General stats ###

** Which burrito is more popular, steak or chicken **




