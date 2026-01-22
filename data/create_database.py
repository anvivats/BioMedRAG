"""
Create SQLite database schema for PubMed documents
"""

import sqlite3
from pathlib import Path


class DatabaseManager:
    """Manages SQLite database creation and initialization."""
    
    def __init__(self, db_path: str = "data/pubmed.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
    
    def connect(self):
        """Connect to database."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        print(f"✓ Connected to database: {self.db_path}")
    
    def create_schema(self):
        """Create database tables and indexes."""
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
        
        # Create indexes for fast lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pmid 
            ON documents(pmid)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_published_date 
            ON documents(published_date)
        """)
        
        # Optional: Full-text search index (FTS5)
        # This allows fast text search within abstracts
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts 
            USING fts5(
                pmid,
                title,
                abstract,
                content='documents',
                content_rowid='pmid'
            )
        """)
        
        # Triggers to keep FTS table in sync
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_ai 
            AFTER INSERT ON documents 
            BEGIN
                INSERT INTO documents_fts(pmid, title, abstract)
                VALUES (new.pmid, new.title, new.abstract);
            END
        """)
        
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_ad 
            AFTER DELETE ON documents 
            BEGIN
                DELETE FROM documents_fts WHERE pmid = old.pmid;
            END
        """)
        
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_au 
            AFTER UPDATE ON documents 
            BEGIN
                UPDATE documents_fts 
                SET title = new.title, abstract = new.abstract
                WHERE pmid = new.pmid;
            END
        """)
        
        self.conn.commit()
        print("✓ Database schema created successfully")
        print("  Tables: documents, documents_fts")
        print("  Indexes: idx_pmid, idx_published_date")
    
    def insert_document(
        self,
        pmid: int,
        title: str,
        abstract: str,
        journal: str = None,
        published_date: str = None,
        authors: str = None,
        mesh_terms: str = None
    ):
        """
        Insert a single document.
        
        Args:
            pmid: PubMed ID
            title: Article title
            abstract: Article abstract
            journal: Journal name (optional)
            published_date: Publication date (optional)
            authors: Comma-separated authors (optional)
            mesh_terms: Comma-separated MeSH terms (optional)
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO documents 
            (pmid, title, abstract, journal, published_date, authors, mesh_terms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (pmid, title, abstract, journal, published_date, authors, mesh_terms))
        
        self.conn.commit()
    
    def insert_documents_batch(self, documents: list):
        """
        Insert multiple documents efficiently.
        
        Args:
            documents: List of tuples (pmid, title, abstract, journal, date, authors, mesh)
        """
        cursor = self.conn.cursor()
        
        cursor.executemany("""
            INSERT OR REPLACE INTO documents 
            (pmid, title, abstract, journal, published_date, authors, mesh_terms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, documents)
        
        self.conn.commit()
        print(f"✓ Inserted {len(documents)} documents")
    
    def get_document(self, pmid: int):
        """Retrieve a single document by PMID."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT pmid, title, abstract, journal, published_date, authors, mesh_terms
            FROM documents
            WHERE pmid = ?
        """, (pmid,))
        return cursor.fetchone()
    
    def get_documents_by_pmids(self, pmids: list):
        """Retrieve multiple documents by PMIDs."""
        placeholders = ','.join('?' * len(pmids))
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT pmid, title, abstract, journal, published_date, authors, mesh_terms
            FROM documents
            WHERE pmid IN ({placeholders})
        """, pmids)
        return cursor.fetchall()
    
    def search_fulltext(self, query: str, limit: int = 10):
        """
        Search documents using full-text search.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching documents
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT d.pmid, d.title, d.abstract, d.journal, d.published_date
            FROM documents d
            JOIN documents_fts fts ON d.pmid = fts.pmid
            WHERE documents_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit))
        return cursor.fetchall()
    
    def get_stats(self):
        """Get database statistics."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_docs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT journal) FROM documents WHERE journal IS NOT NULL")
        total_journals = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(published_date), MAX(published_date) FROM documents WHERE published_date IS NOT NULL")
        date_range = cursor.fetchone()
        
        return {
            'total_documents': total_docs,
            'unique_journals': total_journals,
            'date_range': f"{date_range[0]} to {date_range[1]}" if date_range[0] else "N/A"
        }
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("✓ Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def initialize_database(db_path: str = "data/pubmed.db"):
    """
    Initialize a new database with schema.
    
    Args:
        db_path: Path to database file
    """
    print("=" * 60)
    print("Initializing PubMed Database")
    print("=" * 60)
    
    db = DatabaseManager(db_path)
    db.connect()
    db.create_schema()
    
    # Print stats
    stats = db.get_stats()
    print("\nDatabase Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    db.close()
    
    print("\n✓ Database initialization complete!")
    print(f"  Location: {Path(db_path).absolute()}")


if __name__ == "__main__":
    # Create database
    initialize_database()
    
    # Test with sample data
    print("\n" + "=" * 60)
    print("Testing with Sample Data")
    print("=" * 60)
    
    with DatabaseManager("data/pubmed.db") as db:
        # Insert test document
        db.insert_document(
            pmid=12345678,
            title="Sample PubMed Article",
            abstract="This is a test abstract about diabetes and insulin resistance.",
            journal="Test Journal",
            published_date="2024-01-01"
        )
        
        # Retrieve it
        doc = db.get_document(12345678)
        if doc:
            print(f"\n✓ Test document inserted and retrieved:")
            print(f"  PMID: {doc['pmid']}")
            print(f"  Title: {doc['title']}")
            print(f"  Abstract: {doc['abstract'][:50]}...")
        
        # Test full-text search
        results = db.search_fulltext("diabetes")
        print(f"\n✓ Full-text search found {len(results)} results")