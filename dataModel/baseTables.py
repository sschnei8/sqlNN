import query

# Create table to hold input data in our training database
query.query(
"""
CREATE TABLE INPUT_DATA AS (
SELECT * 
FROM read_csv_auto('/Users/iamsam/Desktop/pyProjects/sqlNueralNet/digit-recognizer/train.csv')
);
"""
)

#%%
# [V1 Random WEIGHT INIT] Create table to hold all weights between input and first hidden layer 
query.query(
"""
-- TRUNCATE TABLE INPUT_WEIGHTS;
-- DROP TABLE IF EXISTS INPUT_WEIGHTS;

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
SELECT 0 AS NEURON
UNION 
SELECT NEURON + 1 AS NEURON
FROM TEN_NEURONS
WHERE NEURON < 9
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
# [V2 HE Weight Initialization FIRST LAYER]
query.query(
"""
-- TRUNCATE TABLE INPUT_WEIGHTS_HE_INIT;
-- DROP TABLE IF EXISTS INPUT_WEIGHTS_HE_INIT;

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
SELECT 0 AS NEURON
UNION 
SELECT NEURON + 1 AS NEURON
FROM TEN_NEURONS
WHERE NEURON < 9
)

SELECT * FROM TEN_NEURONS
);

CREATE TABLE INPUT_WEIGHTS_HE_INIT AS (
-- SQRT(6 / N) WHERE N IS THE NUMBER OF INPUTS
WITH VARIANCE AS (
SELECT SQRT(6::FLOAT / COUNT(1)) AS VAR
FROM STAGE_INPUTS
)

-- NOW WE CROSS JOIN INPUTS EXPLODING THE RESULT SET BY 784 ROWS FOR EACH NEURON
-- LASTLY WE ADD OUR WEIGHTS && CREATE OUR TABLE
-- Because RANDOM() PRODUCES VALUES BETWEEN 0 & 1, WE SHIFT TO -1 & 1
SELECT N.NEURON
     , A.INPUT_LAYER
     , ((RANDOM() * 2) - 1) * V.VAR AS WEIGHT -- TRANSFORM TO NEW RANGE -1 -> 1, MULTIPLY BY VAR
FROM STAGE_NUERONS N
CROSS JOIN STAGE_INPUTS A
LEFT JOIN VARIANCE V ON 1=1 -- ADD VAR TO EACH ROW 
);
"""
)

# %%
# STORE A BIAS VALUE FOR EACH OF OUR NEURONS IN OUR FIRST HIDDEN LAYER 

query.query(
"""
-- TRUNCATE TABLE BIAS_CACHE;
-- DROP TABLE IF EXISTS BIAS_CACHE;

CREATE TABLE BIAS_CACHE AS (
WITH RECURSIVE TEN_NEURONS (NEURON) AS ( -- GENERATE TEN NERONS
SELECT 0 AS NEURON
UNION 
SELECT NEURON + 1 AS NEURON
FROM TEN_NEURONS
WHERE NEURON < 9
)

SELECT NEURON
     , 0 AS BIAS  
FROM TEN_NEURONS
);
"""
)
#%%
# CREATING A TABLE TO CACHE THE LABEL & PREDICTION FOR EACH IMAGE
query.query(
"""
-- DROP TABLE IF EXISTS RESULT_CACHE;

CREATE TABLE RESULT_CACHE (
   ID UINTEGER PRIMARY KEY -- ID OF THE IMAGE 
 , LABEL UTINYINT CHECK(LABEL BETWEEN 0 AND 9) -- WHAT THE HAND WRITTEN NUMBER WAS [0 - 9] CAN BE 0 THROUH 9
 , PREDICTION UTINYINT CHECK(PREDICTION BETWEEN 0 AND 9) -- WHAT WE ARE PREDICTING [0 - 9] CAN BE 0 THROUGH 9
);
"""
)

# Demonstrate our constraint preventing invalid values
# query.query(
# """
# INSERT INTO RESULT_CACHE(ID, LABEL, PREDICTION) VALUES
# (1, 0, 22)
# """
# )

# %%

# ------------------------------------------------
# *** FORWARD PASS TO FIRST HIDDEN LAYER ***]
# ------------------------------------------------

# 1. We take out 784 pixels and transpose them 
# 2. We Join them and their associated values to our input layer 
# 3. We Multiply Weight * The newly Joined values and sum for neuron 
# 4. We add our bias 
# 5. We apply our activation function via case statement ReLU for each node 

# 6. going to need some table to cache intermediate results just not sure what this looks like yet 

query.query(
"""
-- INSERT INTO OUR RESULTS TABLE 
BEGIN TRANSACTION;
INSERT INTO RESULT_CACHE(
SELECT ID
     , LABEL
     , NULL AS PREDICTION
FROM INPUT_DATA 
WHERE ID = 1 -- FIRST IMAGE
);
COMMIT;


-- TRUNCATE TABLE FIRST_HIDDEN_LAYER_VALUES;
-- DROP TABLE IF EXISTS FIRST_HIDDEN_LAYER_VALUES;

CREATE TEMP TABLE TRANSPOSED_PIXELS AS (
WITH ONE_IMAGE AS (
SELECT *
FROM INPUT_DATA 
WHERE ID = 1 -- FIRST IMAGE
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

CREATE TABLE FIRST_HIDDEN_LAYER_VALUES AS (
WITH INPUT_TIMES_WEIGHT AS (
SELECT I.NEURON
     , I.INPUT_LAYER
     , I.WEIGHT
     , P.PIXEL_VALUE
     , I.WEIGHT * P.PIXEL_VALUE AS WEIGHTxVALUE
FROM INPUT_WEIGHTS_HE_INIT I
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
            ELSE W.NEWVAL + B.BIAS END AS ACTIVATED_VALUE    
FROM WEIGHTED_SUM W
    LEFT JOIN BIAS_CACHE B ON W.NEURON = B.NEURON

);

-- COPY CHECK123 TO 'output.csv' (HEADER, DELIMITER ',');
"""
)

# %%

# Initialize these weights between the First Hidden Layer and the Output layer 
# Ten Hidden Neurons * Ten Output Neurons = 100 Weights to initialize 
# Going to use HE Weight Initialization again

query.query(
"""
-- TRUNCATE TABLE HIDDEN_WEIGHTS_HE_INIT;
-- DROP TABLE IF EXISTS HIDDEN_WEIGHTS_HE_INIT;

CREATE TEMP TABLE STAGE_INPUTS AS (
WITH RECURSIVE HIDDEN_INPUTS (HIDDEN_LAYER) AS ( -- GENERATE OUR 10 HIDDEN NEURONS
SELECT 0 AS HIDDEN_LAYER
UNION 
SELECT HIDDEN_LAYER + 1 AS HIDDEN_LAYER
FROM HIDDEN_INPUTS
WHERE HIDDEN_LAYER < 9
)

SELECT * FROM HIDDEN_INPUTS
);

CREATE TEMP TABLE STAGE_OUTPUTS AS (
WITH RECURSIVE TEN_OUTPUTS (OUTPUT) AS ( -- GENERATE TEN OUTPUTS
SELECT 0 AS OUTPUT
UNION 
SELECT OUTPUT + 1 AS OUTPUT
FROM TEN_OUTPUTS
WHERE OUTPUT < 9
)

SELECT * FROM TEN_OUTPUTS
);

CREATE TABLE HIDDEN_WEIGHTS_HE_INIT AS (
-- SQRT(6 / N) WHERE N IS THE NUMBER OF INPUTS
WITH VARIANCE AS (
SELECT SQRT(6::FLOAT / COUNT(1)) AS VAR
FROM STAGE_INPUTS
)

-- NOW WE CROSS JOIN INPUTS EXPLODING THE RESULT SET BY 10 ROWS FOR EACH OUTPUT NDOE
-- LASTLY WE ADD OUR WEIGHTS && CREATE OUR TABLE
-- Because RANDOM() PRODUCES VALUES BETWEEN 0 & 1, WE SHIFT TO -1 & 1
SELECT N.OUTPUT
     , A.HIDDEN_LAYER
     , ((RANDOM() * 2) - 1) * V.VAR AS WEIGHT -- TRANSFORM TO NEW RANGE -1 -> 1, MULTIPLY BY VAR
FROM STAGE_OUTPUTS N
CROSS JOIN STAGE_INPUTS A
LEFT JOIN VARIANCE V ON 1=1 -- ADD VAR TO EACH ROW 
);
"""
)

# %%
# STORE A BIAS VALUE FOR EACH OF OUR OUTPUT NODES

query.query(
"""
-- TRUNCATE TABLE OUTPUT_BIAS_CACHE;
-- DROP TABLE IF EXISTS OUTPUT_BIAS_CACHE;

CREATE TABLE OUTPUT_BIAS_CACHE AS (
WITH RECURSIVE TEN_OUTPUTS (OUTPUT) AS ( -- GENERATE TEN OUTPUTS
SELECT 0 AS OUTPUT
UNION 
SELECT OUTPUT + 1 AS OUTPUT
FROM TEN_OUTPUTS
WHERE OUTPUT < 9
)

SELECT OUTPUT
     , 0 AS BIAS  
FROM TEN_OUTPUTS
);
"""
)

#%%
# Table to store Euler's Number "e" for use in Soft Max activation function
query.query(
"""
-- FOR SOFTMAX (THE ACTIVATION FUNCTION IN OUR OUTPUT LAYER), TYPICALLY e IS USED
-- AS FAR AS I CAN TELL e AS A CONSTANT DOES NOT EXIST IN DUCKDB SO WE'LL NEED TO GENERATE THIS OURSELVES 
-- Luckily we can do this via the infinite series by summing 1/N! from N = 0 -> N = Infinity

CREATE TABLE EULERS_NUMBER AS (
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
"""
)


# %% HIDDEN LAYER -> OUTPUT LAYER FORWARD PASS
query.query(
"""

-- TRUNCATE TABLE FINAL_OUTPUTS;
-- DROP TABLE IF EXISTS FINAL_OUTPUTS;

CREATE TABLE FINAL_OUTPUTS AS (
WITH LAYER_ONE_VALUES AS (
SELECT H.OUTPUT
     , H.WEIGHT
     , F.ACTIVATED_VALUE
     , H.WEIGHT * F.ACTIVATED_VALUE AS WEIGHTxVALUE
FROM HIDDEN_WEIGHTS_HE_INIT H
        LEFT JOIN FIRST_HIDDEN_LAYER_VALUES F ON H.OUTPUT = F.NEURON
),

WEIGHTED_SUM AS (
 SELECT OUTPUT
      , SUM(WEIGHTxVALUE) AS WEIGHTED_SUM
 FROM LAYER_ONE_VALUES
 GROUP BY 1
), 

-- Generate our final outputs for this pass
BIAS_ADDITION AS (
SELECT W.OUTPUT
     , W.WEIGHTED_SUM + O.BIAS AS OUTPUT_VALUE
FROM WEIGHTED_SUM W
    LEFT JOIN OUTPUT_BIAS_CACHE O ON W.OUTPUT = O.OUTPUT
),

-- We may run into issues raising e to thepower of a large number 
-- We can normalize the outputs by subtracting the inpout vector by a constant 
-- In this case we'll find the largest element of our vector and substract it from all elements before our activation function
SOFT_MAX AS (
SELECT B.OUTPUT
     , B.OUTPUT_VALUE 
     , EN.e
     , MAX(OUTPUT_VALUE) OVER() AS MAX_OUTPUT
     , SUM(OUTPUT_VALUE) OVER() AS NET_OUTPUT
     , (B.OUTPUT_VALUE - MAX_OUTPUT)::DOUBLE AS FINAL_OUTPUT_FOR_SOFR_MAX
     , EN.e^FINAL_OUTPUT_FOR_SOFR_MAX AS SOFT_MAX_VALS
FROM BIAS_ADDITION B
    LEFT JOIN EULERS_NUMBER EN ON 1=1
)

-- DUCKDB DOESNT HAVE RATIO TO REPORT UNFORTUNATELY
SELECT OUTPUT
     , SOFT_MAX_VALS
     , SUM(SOFT_MAX_VALS) OVER() AS TOTAL_SOFT_MAX_OUTPUT
     , ROUND(SOFT_MAX_VALS / TOTAL_SOFT_MAX_OUTPUT, 6) AS OUTPUT_PERC
FROM SOFT_MAX
);

-- NOW UPDATE RESULT CACHE
-- HARDCODED ID HERE, WILL NEED TO PASS IT IN OUR LOOP

-- IDK HWO TO AVOID THE LOOP BUT AS LONG AS WE AREJUST PASSING THINGS IN MIGHT AS WELL

WITH OUTPUT_PRED AS (
SELECT 1 AS ID 
     , OUTPUT
FROM FINAL_OUTPUTS
QUALIFY ROW_NUMBER() OVER(ORDER BY OUTPUT_PERC DESC) = 1 -- SELECT OUTPUT WITH HIGHEST PERCENTAGE
)

UPDATE RESULT_CACHE R SET PREDICTION = OUTPUT FROM OUTPUT_PRED P WHERE R.ID = P.ID


"""
)
# %%
query.query(
"""
SELECT * FROM RESULT_CACHE
"""
)

# %%
