import duckdb

# Define a function to execute and write results back to the interactive shelll
# Create a DuckDB instance
# duckdb.connect(dbname) creates a connection to a persistent database

def query(sqltext:str) -> None: 
    conn = duckdb.connect("training.db")
    print(conn.execute(sqltext).df())

if __name__ == '__main__':
    query()