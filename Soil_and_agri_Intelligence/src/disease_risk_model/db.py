import psycopg2
from psycopg2.extras import RealDictCursor
import os
import logging

logger = logging.getLogger(__name__)

def get_connection():
    """
    Establish and return a connection to the PostgreSQL database.
    Expects environment variables for configuration.
    Defaults are provided for local testing but should be overridden in production.
    """
    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        database=os.environ.get("DB_NAME", "agri_db"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "postgres"),
        port=os.environ.get("DB_PORT", "5432")
    )
    return conn

def execute_query(query, params=None, fetch=False):
    """
    Execute a single query.
    
    Args:
        query (str): SQL query to execute.
        params (tuple/dict, optional): Parameters to bind to the query.
        fetch (bool): Whether to fetch and return the results.
        
    Returns:
        List of RealDictRow if fetch is True, otherwise None.
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params)
            if fetch:
                return cursor.fetchall()
            conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error execution query: {e}")
        raise e
    finally:
        conn.close()
