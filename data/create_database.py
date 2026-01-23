"""
Create SQLite database schema for PubMed documents
"""

import sqlite3
from pathlib import Path


class DatabaseManager:
    """Manages SQLite database creation and initialization."""

    def __init__(self, db_path: str = "data/pubmed.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None

    def connect(self):
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        print(f"✓ Connected to database: {self.db_path}")

    def create_schema(self):
        if self.conn is None:
            self.connect()

        cursor = self.conn.cursor()

        # Main documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                pmid INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                journal TEXT,
                published_date TEXT,
                authors TEXT,
                mesh_terms TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_published_date
            ON documents(published_date)
        """)

        # ✅ SAFE standalone FTS table (NO content=)
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
            USING fts5(
                pmid,
                title,
                abstract
            )
        """)

        self.conn.commit()
        print("✓ Database schema created successfully")
        print("  Tables: documents, documents_fts")

    def insert_documents_batch(self, documents: list):
        """
        documents: List of tuples
        (pmid, title, abstract, journal, date, authors, mesh)
        """
        cursor = self.conn.cursor()

        cursor.executemany("""
            INSERT OR REPLACE INTO documents
            (pmid, title, abstract, journal, published_date, authors, mesh_terms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, documents)

        cursor.executemany("""
            INSERT INTO documents_fts (pmid, title, abstract)
            VALUES (?, ?, ?)
        """, [(d[0], d[1], d[2]) for d in documents])

        self.conn.commit()
        print(f"✓ Inserted {len(documents)} documents")

    def get_document(self, pmid: int):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT pmid, title, abstract, journal, published_date, authors, mesh_terms
            FROM documents
            WHERE pmid = ?
        """, (pmid,))
        return cursor.fetchone()

    def search_fulltext(self, query: str, limit: int = 10):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT pmid, title, abstract
            FROM documents_fts
            WHERE documents_fts MATCH ?
            LIMIT ?
        """, (query, limit))
        return cursor.fetchall()

    def get_stats(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_docs = cursor.fetchone()[0]

        return {"total_documents": total_docs}

    def close(self):
        if self.conn:
            self.conn.close()
            print("✓ Database connection closed")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def initialize_database(db_path: str = "data/pubmed.db"):
    print("=" * 60)
    print("Initializing PubMed Database")
    print("=" * 60)

    db = DatabaseManager(db_path)
    db.connect()
    db.create_schema()
    db.close()

    print("\n✓ Database initialization complete!")
    print(f"  Location: {Path(db_path).absolute()}")


if __name__ == "__main__":
    initialize_database()
