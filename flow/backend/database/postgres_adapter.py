"""
PostgreSQL Database Adapter for AutoAudit
Day 2.1 - Quick Database Setup

This module provides an async PostgreSQL adapter using asyncpg for high-performance
database operations in the AutoAudit application.
"""

import os
import asyncpg
from contextlib import asynccontextmanager
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class PostgresAdapter:
    """
    PostgreSQL database adapter with connection pooling and async operations.
    """
    
    def __init__(self, database_url: str):
        """
        Initialize the PostgreSQL adapter.
        
        Args:
            database_url: PostgreSQL connection URL
        """
        self.database_url = database_url
        self.pool: Optional[asyncpg.Pool] = None
    
    async def init_pool(self, min_size: int = 5, max_size: int = 10):
        """
        Initialize the connection pool.
        
        Args:
            min_size: Minimum number of connections in the pool
            max_size: Maximum number of connections in the pool
        """
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=min_size,
                max_size=max_size,
                command_timeout=60,
                server_settings={
                    'application_name': 'autoaudit',
                    'jit': 'off'
                }
            )
            logger.info(f"PostgreSQL connection pool initialized with {min_size}-{max_size} connections")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    async def close_pool(self):
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """
        Get a connection from the pool.
        
        Yields:
            asyncpg.Connection: Database connection
        """
        if not self.pool:
            raise RuntimeError("Database pool not initialized. Call init_pool() first.")
        
        async with self.pool.acquire() as connection:
            yield connection
    
    async def execute(self, query: str, *args, timeout: float = None):
        """
        Execute a query without returning results.
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Query timeout in seconds
        
        Returns:
            str: Status message
        """
        async with self.get_connection() as conn:
            return await conn.execute(query, *args, timeout=timeout)
    
    async def fetch_one(self, query: str, *args, timeout: float = None) -> Optional[Dict[str, Any]]:
        """
        Fetch a single row.
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Query timeout in seconds
        
        Returns:
            Optional[Dict[str, Any]]: Single row as a dictionary or None
        """
        async with self.get_connection() as conn:
            row = await conn.fetchrow(query, *args, timeout=timeout)
            return dict(row) if row else None
    
    async def fetch_all(self, query: str, *args, timeout: float = None) -> List[Dict[str, Any]]:
        """
        Fetch all rows.
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Query timeout in seconds
        
        Returns:
            List[Dict[str, Any]]: List of rows as dictionaries
        """
        async with self.get_connection() as conn:
            rows = await conn.fetch(query, *args, timeout=timeout)
            return [dict(row) for row in rows]
    
    # Document-specific methods
    
    async def save_transferencia(self, data: Dict[str, Any]) -> int:
        """Save a transferencia record."""
        query = """
            INSERT INTO transferencias (concepto, fecha_valor, importe, id_documento)
            VALUES ($1, $2, $3, $4)
            RETURNING id
        """
        async with self.get_connection() as conn:
            result = await conn.fetchval(
                query,
                data.get('concepto'),
                data.get('fecha_valor'),
                data.get('importe'),
                data.get('id_documento')
            )
            return result
    
    async def save_factura(self, data: Dict[str, Any]) -> int:
        """Save a factura record."""
        query = """
            INSERT INTO facturas (
                cif_cliente, cliente, id_documento, numero_factura,
                fecha_factura, proveedor, base_imponible, cif_proveedor,
                irpf, iva, total_factura
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING id
        """
        async with self.get_connection() as conn:
            result = await conn.fetchval(
                query,
                data.get('cif_cliente'),
                data.get('cliente'),
                data.get('id_documento'),
                data.get('numero_factura'),
                data.get('fecha_factura'),
                data.get('proveedor'),
                data.get('base_imponible'),
                data.get('cif_proveedor'),
                data.get('irpf'),
                data.get('iva'),
                data.get('total_factura')
            )
            return result
    
    async def save_nomina(self, data: Dict[str, Any]) -> int:
        """Save a nomina record."""
        query = """
            INSERT INTO nominas (
                id_documento, mes, fecha_inicio, fecha_fin,
                cif, trabajador, naf, nif, categoria,
                total_devengos, total_deducciones, liquido_a_percibir
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            RETURNING id
        """
        async with self.get_connection() as conn:
            result = await conn.fetchval(
                query,
                data.get('id_documento'),
                data.get('mes'),
                data.get('fecha_inicio'),
                data.get('fecha_fin'),
                data.get('cif'),
                data.get('trabajador'),
                data.get('naf'),
                data.get('nif'),
                data.get('categoria'),
                data.get('total_devengos'),
                data.get('total_deducciones'),
                data.get('liquido_a_percibir')
            )
            return result
    
    async def save_workflow_execution(self, workflow_id: str, workflow_type: str, 
                                    status: str, started_at: datetime,
                                    metadata: Optional[Dict] = None) -> str:
        """Save workflow execution log."""
        query = """
            INSERT INTO workflow_executions (
                workflow_id, workflow_type, status, started_at, metadata
            )
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
        """
        async with self.get_connection() as conn:
            result = await conn.fetchval(
                query,
                workflow_id,
                workflow_type,
                status,
                started_at,
                json.dumps(metadata) if metadata else None
            )
            return str(result)
    
    async def update_workflow_execution(self, execution_id: str, status: str,
                                      completed_at: Optional[datetime] = None,
                                      error_message: Optional[str] = None):
        """Update workflow execution status."""
        query = """
            UPDATE workflow_executions
            SET status = $2,
                completed_at = $3,
                duration_seconds = EXTRACT(EPOCH FROM ($3 - started_at)),
                error_message = $4
            WHERE id = $1
        """
        async with self.get_connection() as conn:
            await conn.execute(
                query,
                execution_id,
                status,
                completed_at,
                error_message
            )
    
    async def get_document_processing_status(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document processing status."""
        query = """
            SELECT * FROM document_processing_status
            WHERE document_id = $1
        """
        return await self.fetch_one(query, document_id)
    
    async def update_document_processing_status(self, document_id: str, status: str,
                                              **kwargs):
        """Update document processing status."""
        # Build dynamic update query based on provided kwargs
        set_clauses = ["processing_status = $2", "updated_at = CURRENT_TIMESTAMP"]
        values = [document_id, status]
        param_count = 3
        
        for key, value in kwargs.items():
            if key in ['confidence_score', 'extracted_data', 'error_details', 
                      'processing_attempts', 'completed_at']:
                set_clauses.append(f"{key} = ${param_count}")
                if key == 'extracted_data' and isinstance(value, dict):
                    value = json.dumps(value)
                values.append(value)
                param_count += 1
        
        query = f"""
            UPDATE document_processing_status
            SET {', '.join(set_clauses)}
            WHERE document_id = $1
        """
        
        async with self.get_connection() as conn:
            await conn.execute(query, *values)
    
    async def get_app_config(self, config_key: str) -> Optional[Any]:
        """Get application configuration value."""
        query = "SELECT config_value FROM app_config WHERE config_key = $1"
        result = await self.fetch_one(query, config_key)
        if result and result['config_value']:
            return result['config_value']
        return None
    
    async def set_app_config(self, config_key: str, config_value: Any, 
                           description: Optional[str] = None):
        """Set application configuration value."""
        query = """
            INSERT INTO app_config (config_key, config_value, description)
            VALUES ($1, $2, $3)
            ON CONFLICT (config_key)
            DO UPDATE SET 
                config_value = EXCLUDED.config_value,
                updated_at = CURRENT_TIMESTAMP
        """
        async with self.get_connection() as conn:
            await conn.execute(
                query,
                config_key,
                json.dumps(config_value) if not isinstance(config_value, str) else config_value,
                description
            )
    
    async def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            async with self.get_connection() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global database instance
database_url = os.getenv(
    "DATABASE_URL", 
    "postgresql://auditai:password@localhost:5432/auditai"
)
db = PostgresAdapter(database_url)


# Helper function for FastAPI lifespan
async def init_database():
    """Initialize database connection pool."""
    await db.init_pool()
    logger.info("Database initialized")


async def close_database():
    """Close database connection pool."""
    await db.close_pool()
    logger.info("Database closed")