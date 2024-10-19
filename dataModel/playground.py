#%%
import duckdb
import query
from dotenv import load_dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt

#path to training data stored in env variaable
training_data_location = os.getenv('train_csv')

#%%

#print(conn.execute(
#"""
#SELECT * FROM read_csv('/Users/iamsam/Desktop/pyProjects/sqlNueralNet/digit-recognizer/train.csv') LIMIT 4
#"""
#).df()
#)


# %%
query.query(
'SELECT COUNT(1) FROM INPUT_DATA'
)

# %%

# neuron_id        | input_id
# [1, 2, 3,... 10] | [1, 2, 3,... 784]
query.query(
"""SELECT generate_series(1, 10) AS neuron_id
   , generate_series(1, 784) AS input_id"""
)

#Random(): Returns a random number x in the range 0.0 <= x < 1.0.
#%%
query.query(
"""
CREATE TEMP TABLE STAGE_INPUTS AS (
WITH RECURSIVE ALL_INPUTS (INPUT_LAYER) AS ( -- GENERATE 784 INPUT PIXELS
SELECT 0 AS INPUT_LAYER
UNION 
SELECT INPUT_LAYER + 1 AS INPUT_LAYER
FROM ALL_INPUTS
WHERE INPUT_LAYER < 783
)
SELECT * FROM ALL_INPUTS
);

CREATE TEMP TABLE STAGE_NUERONS AS (
WITH RECURSIVE TEN_NEURONS (NEURON) AS ( -- GENERATE TEN NERONS
SELECT 1 AS NEURON
UNION 
SELECT NEURON + 1 AS NEURON
FROM TEN_NEURONS
WHERE NEURON < 10
)

SELECT * FROM TEN_NEURONS
);

-- NOW WE CROSS JOIN INPUTS EXPLODING THE RESULT SET BY 784 ROWS FOR EACH NEURON
-- LASTLY WE ADD OUR WEIGHTS && CREATE OUR TABLE
CREATE TABLE INPUT_WEIGHTS AS (
SELECT N.NEURON
     , A.INPUT_LAYER
     , RANDOM() AS WEIGHT
FROM STAGE_NUERONS N
CROSS JOIN STAGE_INPUTS A
);
"""
)

# %%
query.query(
"""
WITH RECURSIVE ALL_INPUTS (INPUT_LAYER) AS (
SELECT 0 AS INPUT_LAYER
UNION 
SELECT INPUT_LAYER + 1 AS INPUT_LAYER
FROM ALL_INPUTS
WHERE INPUT_LAYER < 783
)
SELECT * FROM ALL_INPUTS
"""
)
#%%
#query.query(
#"""
#WITH RECURSIVE RAND_CTE (NUM, RAND_FLOAT) AS (
#SELECT 0 AS NUM
#     , 0.00 AS RAND_FLOAT
#UNION 
#SELECT NUM + 1 AS NUM
#     , RANDOM() AS RAND_FLOAT
#FROM RAND_CTE
#WHERE NUM < 100000
#)
#
#SELECT RAND_FLOAT, COUNT(1) FROM RAND_CTE GROUP BY 1
#"""
#)

#View the distribution of random()
#Seems to approach a uniform distribution outside of 0 and 1 
conn = duckdb.connect("connections.db")
result = conn.execute("""
WITH RECURSIVE RAND_CTE (NUM, RAND_FLOAT) AS (
SELECT 0 AS NUM
     , 0.00 AS RAND_FLOAT
UNION 
SELECT NUM + 1 AS NUM
     , RANDOM() AS RAND_FLOAT
FROM RAND_CTE
WHERE NUM < 100000
)

SELECT RAND_FLOAT, COUNT(1) AS TOTAL FROM RAND_CTE GROUP BY 1
""").df()

# Example: Bar chart
result.plot.bar(x='RAND_FLOAT', y='TOTAL')
plt.show()

# %%
# 1. We take out 784 pixels and transpose them 
# 2. We Join them and their associated values to our input layer 
# 3. We Multiply Weight * THe newly Joined values and sum for neuron 
# 4. We add our bias 
# 5. We peerform our case statement RELU for each node 

# 6. going to need some table to cache intermediate results just not sure what this looks like yet 

query.query(
"""
CREATE TEMP TABLE TRANSPOSED_PIXELS AS (
WITH ONE_IMAGE AS (
SELECT *
FROM INPUT_DATA 
WHERE ID  = 1 -- FIRST IMAGE
),

TRANSPOSE AS (
UNPIVOT (FROM ONE_IMAGE) ON COLUMNS(*) INTO NAME PIXEL VALUE PIXEL_VALUE
)

SELECT PIXEL
     , PIXEL_VALUE
     , LTRIM(PIXEL, 'pixel') AS PIXEL_ID
FROM TRANSPOSE
WHERE PIXEL NOT IN ('id','label')
);

WITH INPUT_TIMES_WEIGHT AS (
SELECT I.NEURON
     , I.INPUT_LAYER
     , I.WEIGHT
     , P.PIXEL_VALUE
     , I.WEIGHT * P.PIXEL_VALUE AS WEIGHTxVALUE
FROM INPUT_WEIGHTS I
    LEFT JOIN TRANSPOSED_PIXELS P ON INPUT_LAYER = PIXEL_ID
), 

WEIGHTED_SUM AS (
SELECT T.NEURON
     , SUM(WEIGHTxVALUE) AS NEWVAL 
FROM INPUT_TIMES_WEIGHT T
GROUP BY 1
)

-- IMPLEMENT RELU
SELECT W.NEURON
     , CASE WHEN W.NEWVAL + B.BIAS < 0 THEN 0 
            ELSE W.NEWVAL + B.BIAS END AS ACTIVATED_WEIGHT     
FROM WEIGHTED_SUM W
    LEFT JOIN BIAS_CACHE B ON W.NEURON = B.NEURON

-- COPY CHECK123 TO 'output.csv' (HEADER, DELIMITER ',');
"""
)


# %%
query.query(
"""
-- STORE A BIAS VALUE FOR EACH OF OUR NUERONS IN OUR FIRST HIDDEN LAYER 
WITH RECURSIVE TEN_NEURONS (NEURON) AS ( -- GENERATE TEN NERONS
SELECT 1 AS NEURON
UNION 
SELECT NEURON + 1 AS NEURON
FROM TEN_NEURONS
WHERE NEURON < 10
)

SELECT NEURON
     , 0 AS BIAS  
FROM TEN_NEURONS
"""
)

# %%