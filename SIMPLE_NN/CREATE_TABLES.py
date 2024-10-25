#%% 
import duckdb
import pandas

def query(sqltext:str) -> None: 
    conn = duckdb.connect("tiny_nn_in_sql.db")
    print(conn.execute(sqltext).df())

#%%
query(
"""
-- CREATE SCHEMA SIMPLE_NN;

DROP TABLE IF EXISTS SIMPLE_NN.L1_WEIGHTS;
CREATE TABLE SIMPLE_NN.L1_WEIGHTS (
INPUT_NODE INTEGER,
HIDDEN_LAYER INTEGER,
INPUT_VALUE DOUBLE,
WEIGHT DOUBLE
);

INSERT INTO SIMPLE_NN.L1_WEIGHTS(INPUT_NODE, HIDDEN_LAYER, INPUT_VALUE, WEIGHT)
VALUES
(1, 1, .05, .15),
(1, 2, .05, .25),
(2, 1, .10, .20),
(2, 2, .10, .30)
;


DROP TABLE IF EXISTS SIMPLE_NN.L1_BIAS;
CREATE TABLE SIMPLE_NN.L1_BIAS (
HIDDEN_LAYER INTEGER,
BIAS DOUBLE
);

INSERT INTO SIMPLE_NN.L1_BIAS(HIDDEN_LAYER, BIAS)
VALUES
(1, .35),
(2, .35)
;


DROP TABLE IF EXISTS  SIMPLE_NN.L2_WEIGHTS;
CREATE TABLE SIMPLE_NN.L2_WEIGHTS (
HIDDEN_LAYER INTEGER,
OUTPUT_NODE INTEGER,
HIDDEN_VALUE DOUBLE,
WEIGHT DOUBLE
);

INSERT INTO SIMPLE_NN.L2_WEIGHTS(HIDDEN_LAYER, OUTPUT_NODE, HIDDEN_VALUE, WEIGHT)
VALUES
(1, 1, NULL, .40),
(1, 2, NULL, .45),
(2, 1, NULL, .50),
(2, 2, NULL, .55)
;


DROP TABLE IF EXISTS  SIMPLE_NN.L2_BIAS;
CREATE TABLE SIMPLE_NN.L2_BIAS (
OUTPUT_NODE INTEGER,
BIAS DOUBLE
);

INSERT INTO SIMPLE_NN.L2_BIAS(OUTPUT_NODE, BIAS)
VALUES
(1, .6),
(2, .6)
;

DROP TABLE IF EXISTS  SIMPLE_NN.EULERS_NUMBER;
CREATE TABLE SIMPLE_NN.EULERS_NUMBER AS (
WITH RECURSIVE INF_SERIES (N, FACTORIAL, TERM, NSUM) AS (

SELECT 0 AS N
     , 1::UBIGINT AS FACTORIAL -- UNSIGNED EIGHT-BYTE INTEGER
     , 1.000 AS TERM
     , 1::DOUBLE AS NSUM
UNION ALL
SELECT N + 1 AS N
     , FACTORIAL * (N + 1) AS FACTORIAL
     , 1.0 / (FACTORIAL * (N + 1)) AS TERM
     , NSUM + (1.0 / (FACTORIAL * (N + 1))) AS NSUM
FROM INF_SERIES
WHERE N < 14 -- Gets us a DOUBLE fairly close to Euler's Number
)

SELECT NSUM AS e
FROM INF_SERIES
QUALIFY ROW_NUMBER() OVER(ORDER BY N DESC) = 1
);


DROP TABLE IF EXISTS SIMPLE_NN.RESULT_CACHE;
CREATE TABLE SIMPLE_NN.RESULT_CACHE (
INSERT_TIME TIMESTAMP,
OUTPUT_NODE INTEGER,
PREDICTED_VALUE DOUBLE,
ACTUAL_VALUE DOUBLE,
);


DROP TABLE IF EXISTS  SIMPLE_NN.L2_WEIGHT_CACHE;
CREATE TABLE SIMPLE_NN.L2_WEIGHT_CACHE (
INSERT_TIME TIMESTAMP,
HIDDEN_LAYER INTEGER,
OUTPUT_NODE INTEGER,
HIDDEN_VALUE DOUBLE,
WEIGHT DOUBLE,
UPDATED_WEIGHT DOUBLE
);


DROP TABLE IF EXISTS  SIMPLE_NN.L1_WEIGHT_CACHE;
CREATE TABLE SIMPLE_NN.L1_WEIGHT_CACHE (
INSERT_TIME TIMESTAMP,
INPUT_NODE INTEGER,
HIDDEN_LAYER INTEGER,
INPUT_VALUE DOUBLE,
WEIGHT DOUBLE,
UPDATED_WEIGHT DOUBLE
);
"""
)

# %%
from Attempt6 import query1, query2, query3, query4, query5, query6

for i in range(1,1500):
    query(query1)
    query(query2)
    query(query3)
    query(query4)
    query(query5)
    query(query6)

#%%
query(
"""
SELECT INSERT_TIME
     , HIDDEN_LAYER
     , HIDDEN_VALUE
FROM SIMPLE_NN.L2_WEIGHT_CACHE
"""
)

#%%

#Prediction over time
import matplotlib.pyplot as plt

conn = duckdb.connect("tiny_nn_in_sql.db")
df1 = conn.execute("""
SELECT * 
FROM SIMPLE_NN.RESULT_CACHE
""").df()

print(df1['OUTPUT_NODE'].unique())

for output_type in df1['OUTPUT_NODE'].unique():
    mask = df1['OUTPUT_NODE'] == output_type
    plt.plot(df1[mask]['INSERT_TIME'],
             df1[mask]['PREDICTED_VALUE'],
             label=f'Predicted ({output_type})',
             linestyle='-',
             marker='.')
    #plt.plot(df1[mask]['INSERT_TIME'],
    #         df1[mask]['ACTUAL_VALUE'],
    #         label=f'Actual ({output_type})',
    #         linestyle='--',
    #         marker='.')

# %%

#Weights (1-4) over time
import matplotlib.pyplot as plt

conn = duckdb.connect("tiny_nn_in_sql.db")
df1 = conn.execute("""
SELECT INSERT_TIME
     , CASE WHEN INPUT_NODE = 1 AND HIDDEN_LAYER = 1 THEN 'W1'   
            WHEN INPUT_NODE = 1 AND HIDDEN_LAYER = 2 THEN 'W2'   
            WHEN INPUT_NODE = 2 AND HIDDEN_LAYER = 1 THEN 'W3'   
            WHEN INPUT_NODE = 2 AND HIDDEN_LAYER = 2 THEN 'W4'  
            END AS WEIGHT_NAME
     , UPDATED_WEIGHT  
FROM SIMPLE_NN.L1_WEIGHT_CACHE
""").df()

for weight in df1['WEIGHT_NAME'].unique():
    mask = df1['WEIGHT_NAME'] == weight
    plt.plot(df1[mask]['INSERT_TIME'],
             df1[mask]['UPDATED_WEIGHT'],
             label=f'Weight ({weight})',
             linestyle='-',
             marker='.')

# %%
#Weights (5-8) over time
import matplotlib.pyplot as plt

conn = duckdb.connect("tiny_nn_in_sql.db")
df1 = conn.execute("""
SELECT INSERT_TIME
     , CASE WHEN HIDDEN_LAYER = 1 AND OUTPUT_NODE = 1 THEN 'W5'   
            WHEN HIDDEN_LAYER = 1 AND OUTPUT_NODE = 2 THEN 'W6'   
            WHEN HIDDEN_LAYER = 2 AND OUTPUT_NODE = 1 THEN 'W7'   
            WHEN HIDDEN_LAYER = 2 AND OUTPUT_NODE = 2 THEN 'W8'  
            END AS WEIGHT_NAME
     , UPDATED_WEIGHT  
FROM SIMPLE_NN.L2_WEIGHT_CACHE
--WHERE WEIGHT_NAME IN ('W8')
""").df()

for weight in df1['WEIGHT_NAME'].unique():
    mask = df1['WEIGHT_NAME'] == weight
    plt.plot(df1[mask]['INSERT_TIME'],
             df1[mask]['UPDATED_WEIGHT'],
             label=f'Weight ({weight})',
             linestyle='-',
             marker='.')

# %%
#Hidden Layer Values over time 
import matplotlib.pyplot as plt

conn = duckdb.connect("tiny_nn_in_sql.db")
df1 = conn.execute("""
SELECT INSERT_TIME
     , HIDDEN_LAYER
     , HIDDEN_VALUE
FROM SIMPLE_NN.L2_WEIGHT_CACHE
""").df()

for layer in df1['HIDDEN_LAYER'].unique():
    mask = df1['HIDDEN_LAYER'] == layer
    plt.plot(df1[mask]['INSERT_TIME'],
             df1[mask]['HIDDEN_VALUE'],
             label=f'layer ({layer})',
             linestyle='-',
             marker='.')

#%%
query(
"""
SELECT INSERT_TIME
     , CASE WHEN HIDDEN_LAYER = 1 AND OUTPUT_NODE = 1 THEN 'W5'   
            WHEN HIDDEN_LAYER = 1 AND OUTPUT_NODE = 2 THEN 'W6'   
            WHEN HIDDEN_LAYER = 2 AND OUTPUT_NODE = 1 THEN 'W7'   
            WHEN HIDDEN_LAYER = 2 AND OUTPUT_NODE = 2 THEN 'W8'  
            END AS WEIGHT_NAME
     , UPDATED_WEIGHT  
FROM SIMPLE_NN.L2_WEIGHT_CACHE
WHERE WEIGHT_NAME IN ('W5', 'W7')
ORDER BY INSERT_TIME DESC LIMIT 10
"""
)
# %%

query(
"""
SELECT *
FROM SIMPLE_NN.RESULT_CACHE
ORDER BY INSERT_TIME DESC LIMIT 100
"""
)
# %%
query(
"""
PRAGMA database_size;
-- CHECKPOINT tiny_nn_in_sql;

""")

# %%
