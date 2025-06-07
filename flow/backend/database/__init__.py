"""
Database module for AutoAudit.

This module provides database adapters and utilities for the application.
"""

from .postgres_adapter import (
    PostgresAdapter,
    db,
    init_database,
    close_database
)

__all__ = [
    'PostgresAdapter',
    'db',
    'init_database',
    'close_database'
]