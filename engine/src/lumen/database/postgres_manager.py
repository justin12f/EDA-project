import os
import psycopg2
from typing import Any

class PostgresManager:
    """Manages connections and data loading for PostgreSQL."""
    
    def __init__(self):
        self.host = os.getenv("POSTGRES_HOST", "localhost")
        self.port = os.getenv("POSTGRES_PORT", "5432")
        self.user = os.getenv("POSTGRES_USER", "admin")
        self.password = os.getenv("POSTGRES_PASSWORD", "admin")
        self.db = os.getenv("POSTGRES_DB", "eda_db")
        self._conn_uri = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"
        self._jdbc_url = f"jdbc:postgresql://{self.host}:{self.port}/{self.db}"

    def get_connection(self):
        """Returns a psycopg2 connection."""
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=self.db
        )
        
    def write_dataframe(self, df: Any, table_name: str, backend: str, if_exists: str = "replace"):
        """Write a dataframe to PostgreSQL based on its backend."""
        if backend == "pandas":
            from sqlalchemy import create_engine
            engine = create_engine(self._conn_uri)
            df.to_sql(table_name, engine, if_exists=if_exists, index=False)
            
        elif backend == "polars":
            # Polars native db writer
            df.write_database(
                table_name=table_name,
                connection=self._conn_uri,
                if_table_exists=if_exists
            )
            
        elif backend == "spark":
            # PySpark JDBC write
            mode = "overwrite" if if_exists == "replace" else "append"
            df.write.format("jdbc") \
                .option("url", self._jdbc_url) \
                .option("dbtable", table_name) \
                .option("user", self.user) \
                .option("password", self.password) \
                .option("driver", "org.postgresql.Driver") \
                .mode(mode) \
                .save()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def execute_query(self, query: str):
        """Execute a raw SQL query (e.g. DDL statements)."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
            conn.commit()
