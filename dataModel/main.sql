----------------------------------------------------------------------------------------------------
-- Create table for input data (assuming 784 input features)
----------------------------------------------------------------------------------------------------
CREATE TABLE input_data AS
SELECT * FROM read_csv_auto('path/to/your/train.csv');

-- Add an ID column to input data 
CREATE SEQUENCE id_sequence START 1; -- CREATE A SEQUENCE STARTING AT 1 TO ADD AN ID COLUMN TO OUR TRAINING DATA
ALTER TABLE INPUT_DATA ADD COLUMN id INTEGER DEFAULT nextval('id_sequence');
SELECT * FROM INPUT_DATA LIMIT 10; 

----------------------------------------------------------------------------------------------------
-- We need to start with some random initial weight values between 0 and 1
-- https://www.deeplearning.ai/ai-notes/initialization/index.html
-- https://proceedings.neurips.cc/paper/2015/hash/ae0eb3eed39d2bcef4622b2499a05fe6-Abstract.html
-- Weight initialization is an interesting topic and this is certinaly an extremely basic approach

-- https://wandb.ai/sauravmaheshkar/initialization/reports/A-Gentle-Introduction-To-Weight-Initialization-for-Neural-Networks--Vmlldzo2ODExMTg
-- ReLU is typically defined as f(x)=max(0,x)f(x) = max(0, x)
-- f(x)=max(0,x), notice that this function does not have a zero mean.
-- ReLU zeros out negative values, halving variance
-- HE Initialization: http://arxiv.org/pdf/1502.01852
-- Becasue we are drawing from a uniform distribution we need to adjust to 6 instead of 2
-- https://www.pinecone.io/learn/weight-initialization/


-- In our first hidden layer we are going to have 10 nuerons 
-- Each Nueron will have 784 inputs one for each pixel in the image 
-- Our table will contain 7840 rows, 784 for each of our ten nuerons and the associated weight 
----------------------------------------------------------------------------------------------------
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
     , RANDOM() AS WEIGHT -- Random Uniform initialization
FROM STAGE_NUERONS N
CROSS JOIN STAGE_INPUTS A
);

---------------------------------------------------------------------------------------------
-- *** A NOTE ON BIAS ***
-- It is possible and common to initialize the biases to be zero
-- since the asymmetry breaking is provided by the small random numbers in the weights.
---------------------------------------------------------------------------------------------
CREATE TABLE BIAS_CACHE AS (
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
);


----------------------------------------------------------------------------------------------------
--
----------------------------------------------------------------------------------------------------
CREATE TABLE weights_hidden_output AS
SELECT
    output_id,
    ARRAY_AGG(random() * 0.01) AS weights
FROM
    (SELECT generate_series(1, 10) AS output_id,
     generate_series(1, 30) AS hidden_id)
GROUP BY output_id;


-- Function for ReLU activation
CREATE FUNCTION relu(x DOUBLE) RETURNS DOUBLE AS $$
    SELECT CASE WHEN x > 0 THEN x ELSE 0 END
$$;
-- Function for softmax activation
CREATE FUNCTION softmax(x DOUBLE[]) RETURNS DOUBLE[] AS $$
    SELECT array_map(e, e / sum(e))
    FROM (SELECT array_map(x, exp(unnest)) AS e)
$$;
-- Forward pass
CREATE TABLE forward_pass AS
WITH hidden_layer AS (
    SELECT
        id,
        array_map(
            array_zip(
                weights_input_hidden.weights,
                array_slice(struct_pack(input_data.*), 2, 784)
            ),
            relu(sum(element_1 * element_2))
        ) AS hidden_output
    FROM input_data, weights_input_hidden
),
output_layer AS (
    SELECT
        id,
        softmax(
            array_map(
                array_zip(
                    weights_hidden_output.weights,
                    hidden_layer.hidden_output
                ),
                sum(element_1 * element_2)
            )
        ) AS output
    FROM hidden_layer, weights_hidden_output
)
SELECT * FROM output_layer;
-- Calculate loss (cross-entropy)
CREATE TABLE loss AS
SELECT
    AVG(-LOG(output[label + 1])) AS cross_entropy_loss
FROM
    forward_pass
JOIN
    input_data ON forward_pass.id = input_data.id;
-- Backpropagation and weight update (simplified)
-- Note: This is a very simplified version and doesn't include all necessary steps
UPDATE weights_hidden_output
SET weights = array_map(
    array_zip(weights, gradient),
    weights - 0.01 * gradient
)
FROM (
    -- Calculate gradient (simplified)
    SELECT
        output_id,
        ARRAY_AGG((predicted - actual) * hidden_output) AS gradient
    FROM
        forward_pass,
        input_data,
        generate_series(1, 10) AS output_id
    GROUP BY output_id
) gradient_calc
WHERE weights_hidden_output.output_id = gradient_calc.output_id;
-- Similar update for weights_input_hidden (omitted for brevity)
-- Prediction on test data
CREATE TABLE predictions AS
WITH hidden_layer AS (
    -- Similar to forward pass
),
output_layer AS (
    -- Similar to forward pass
)
SELECT
    id,
    array_position(output, array_max(output)) - 1 AS predicted_label
FROM output_layer;
-- Create submission file
COPY (
    SELECT
        id AS ImageId,
        predicted_label AS Label
    FROM predictions
    ORDER BY id
) TO 'submission.csv' WITH (HEADER);



