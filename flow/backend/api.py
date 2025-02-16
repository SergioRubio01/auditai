# Copyright 2025-present Sergio García Arrojo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from fastapi import FastAPI, HTTPException, UploadFile, File, Response, BackgroundTasks, Request
from typing import List, Optional, Dict, Any, Union, Literal
import uvicorn
from datetime import datetime, timedelta
import sqlite3
import json
import logging
import os
import shutil
import zipfile
from pathlib import Path
from .models import Transferencia, Tarjeta, ImageResponse, TarjetaRow, Factura, FacturaRow, Nomina, NominaRow
from .utils import encode_image
from . import process_single_image
from urllib.parse import unquote
import pandas as pd
from io import BytesIO
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import asyncio
import multiprocessing
import re
from .workflowmanager import WorkflowManager
from functools import lru_cache
from .utils import PDFConverter
import aiofiles
import mimetypes
from pdf2image import convert_from_bytes
import time
import math
from fastapi.responses import JSONResponse

# Ensure environment is loaded before any other imports
from . import load_environment
load_environment()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AutoAudit API",debug=True)

# Database setup
DB_PATH = Path(os.getenv("DB_PATH","./database/database.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Add near the top of the file with other path definitions
IMAGES_DIR = Path(os.getenv("IMAGE_INPUT_DIR", "./Images"))
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Add this near the top where other constants are defined
MAX_WORKERS = min(multiprocessing.cpu_count(), 6)
process_pool = None

# Add these constants
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Add these constants near other constants
RATE_LIMIT_WINDOW = 60  # 1 minute window
BASE_RATE_LIMIT = 1000  # requests per minute
MAX_BACKOFF = 3600  # maximum backoff in seconds

# Initialize PDF converter
pdf_converter = PDFConverter()

# Add this class before the app initialization
class RateLimiter:
    def __init__(self):
        self.requests = {}  # {ip: [(timestamp, count)]}
        self.backoff_times = {}  # {ip: (retry_after, violation_count)}

    def calculate_backoff(self, violations: int) -> float:
        """Calculate exponential backoff time based on number of violations"""
        backoff = min(math.pow(2, violations) - 1, MAX_BACKOFF)
        return backoff

    async def check_rate_limit(self, ip: str) -> tuple[bool, float]:
        current_time = time.time()
        
        # Check if IP is in backoff period
        if ip in self.backoff_times:
            retry_after, violations = self.backoff_times[ip]
            if current_time < retry_after:
                return False, retry_after - current_time
            else:
                # Reset backoff if window has passed
                del self.backoff_times[ip]

        # Clean old requests
        if ip in self.requests:
            self.requests[ip] = [
                (ts, count) for ts, count in self.requests[ip]
                if current_time - ts < RATE_LIMIT_WINDOW
            ]

        # Initialize if new IP
        if ip not in self.requests:
            self.requests[ip] = []

        # Calculate current request count
        total_requests = sum(count for _, count in self.requests[ip])

        # If within limit, add request
        if total_requests < BASE_RATE_LIMIT:
            if self.requests[ip] and self.requests[ip][-1][0] == current_time:
                self.requests[ip][-1] = (current_time, self.requests[ip][-1][1] + 1)
            else:
                self.requests[ip].append((current_time, 1))
            return True, 0

        # Calculate violation count and backoff
        violations = (total_requests // BASE_RATE_LIMIT)
        backoff_time = self.calculate_backoff(violations)
        retry_after = current_time + backoff_time
        self.backoff_times[ip] = (retry_after, violations)

        return False, backoff_time

# Initialize rate limiter with the app
rate_limiter = RateLimiter()

# Replace the existing rate limit middleware with this new one
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    
    # Skip rate limiting for certain endpoints if needed
    if request.url.path in ["/docs", "/openapi.json", ".xlsx"]:
        return await call_next(request)
    
    allowed, wait_time = await rate_limiter.check_rate_limit(client_ip)
    
    if not allowed:
        retry_after = datetime.utcnow() + timedelta(seconds=wait_time)
        headers = {
            "Retry-After": retry_after.strftime("%a, %d %b %Y %H:%M:%S GMT"),
            "X-RateLimit-Reset": str(int(wait_time)),
            "X-RateLimit-Limit": str(BASE_RATE_LIMIT),
        }
        
        return JSONResponse(
            status_code=429,
            headers=headers,
            content={
                "error": "Too many requests",
                "retry_after_seconds": round(wait_time, 1),
                "message": f"Please wait {round(wait_time, 1)} seconds before retrying"
            }
        )
    
    return await call_next(request)

def init_db():
    """Initialize SQLite database with required tables"""
    with sqlite3.connect(DB_PATH) as conn:
        # Create pagos table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS facturas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cif_cliente TEXT,
                cliente TEXT,
                id_documento TEXT,
                numero_factura TEXT,
                fecha_factura TEXT,
                proveedor TEXT,
                base_imponible TEXT,
                cif_proveedor TEXT,
                irpf TEXT,
                iva TEXT,
                total_factura TEXT,
                fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.execute("""
                CREATE TABLE IF NOT EXISTS pagos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    id_documento TEXT,
                    tipo_documento TEXT NOT NULL,
                    fecha_valor TEXT,
                    ordenante TEXT,
                    beneficiario TEXT,
                    concepto TEXT,
                    importe TEXT,
                    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        
        conn.execute("""
                CREATE TABLE IF NOT EXISTS nominas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    id_documento TEXT,
                    mes TEXT,
                    fecha_inicio TEXT,
                    fecha_fin TEXT,
                    cif TEXT,
                    trabajador TEXT,
                    naf TEXT,
                    nif TEXT,
                    categoria TEXT,
                    antiguedad TEXT,
                    contrato TEXT,
                    total_devengos TEXT,
                    total_deducciones TEXT,
                    absentismos TEXT,
                    bc_teorica TEXT,
                    prorrata TEXT,
                    bc_con_complementos TEXT,
                    total_seg_social TEXT,
                    bonificaciones_ss_trabajador TEXT,
                    total_retenciones TEXT,
                    total_retenciones_ss TEXT,
                    liquido_a_percibir TEXT,
                    a_abonar TEXT,
                    total_cuota_empresarial TEXT,
                    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS facturas_tabla (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concepto TEXT,
                unidades TEXT,
                importe TEXT,
                id_documento TEXT
                fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pagos_tabla (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concepto TEXT,
                fecha_valor TEXT,
                importe TEXT,
                id_documento TEXT
                fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS nominas_tabla (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                descripcion TEXT,
                importe_unidad TEXT,
                unidad TEXT,
                devengos TEXT,
                deducciones TEXT,
                id_documento TEXT,
                fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create processed_images table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_image_id INTEGER NOT NULL,
                filename TEXT NOT NULL UNIQUE,
                content BLOB NOT NULL,
                size INTEGER NOT NULL,
                process_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (original_image_id) REFERENCES images(id)
            )
        """)
        
        # Create separate tables for facturas and pagos images
        conn.execute("""
            CREATE TABLE IF NOT EXISTS images_facturas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                content BLOB NOT NULL,
                size INTEGER NOT NULL,
                mime_type TEXT NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS images_pagos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                content BLOB NOT NULL,
                size INTEGER NOT NULL,
                mime_type TEXT NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS images_nominas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                content BLOB NOT NULL,
                size INTEGER NOT NULL,
                mime_type TEXT NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()

@app.on_event("startup")
async def startup_event():
    """Initialize database and process pool on startup with retry logic"""
    global process_pool
    init_db()
    
    for attempt in range(MAX_RETRIES):
        try:
            process_pool = ProcessPoolExecutor(max_workers=MAX_WORKERS)
            logger.info(f"Process pool initialized with {MAX_WORKERS} workers")
            break
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                logger.error(f"Failed to initialize process pool after {MAX_RETRIES} attempts: {e}")
                raise
            logger.warning(f"Process pool initialization attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(RETRY_DELAY)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup process pool on shutdown"""
    if process_pool:
        process_pool.shutdown()

@app.post("/transferencias/", response_model=Transferencia)
async def create_transferencia(transferencia: Transferencia):
    """Create a new transferencia in the database"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO pagos
                (id_documento, tipo_documento, fecha_valor, ordenante, beneficiario, concepto, importe)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                transferencia.DOCUMENTO, 
                transferencia.TIPO,
                transferencia.FECHA_VALOR,
                transferencia.ORDENANTE,
                transferencia.BENEFICIARIO,
                transferencia.CONCEPTO,
                transferencia.IMPORTE
            ))
            
            inserted_id = cursor.lastrowid
            conn.commit()
            
            # Fetch inserted record
            cursor.execute("""
                SELECT id, id_documento, tipo_documento, fecha_valor, ordenante, beneficiario, 
                       concepto, importe, fecha_creacion
                FROM pagos 
                WHERE id = ?
            """, (inserted_id,))
            
            row = cursor.fetchone()
            return Transferencia(
                DOCUMENTO=row[1],
                TIPO=row[2],
                FECHA_VALOR=row[3],
                ORDENANTE=row[4],
                BENEFICIARIO=row[5],
                CONCEPTO=row[6],
                IMPORTE=row[7]
            )
            
    except Exception as e:
        logger.error(f"Error creating transferencia: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transferencias/", response_model=List[Transferencia])
async def get_transferencias(tipo_documento: Optional[str] = None):
    """Obtener todas las transferencias, filtradas opcionalmente por tipo_documento"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            if tipo_documento:
                cursor.execute("""
                    SELECT id, id_documento, tipo_documento, fecha_valor, ordenante, beneficiario, 
                           concepto, importe, fecha_creacion
                    FROM pagos 
                    WHERE tipo_documento = ?
                    ORDER BY fecha_creacion DESC
                """, (tipo_documento,))
            else:
                cursor.execute("""
                    SELECT id, tipo_documento, fecha_valor, ordenante, beneficiario, 
                           concepto, importe, fecha_creacion
                    FROM pagos
                    ORDER BY fecha_creacion DESC
                """)
            
            results = []
            for row in cursor.fetchall():
                results.append(Transferencia(
                    DOCUMENTO=row[1],
                    TIPO=row[2],
                    FECHA_VALOR=row[3],
                    ORDENANTE=row[4],
                    BENEFICIARIO=row[5],
                    CONCEPTO=row[6],
                    IMPORTE=row[7]
                ))
            
            return results
            
    except Exception as e:
        logger.error(f"Error fetching transferencias: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tarjetas/", response_model=Tarjeta)
async def create_tarjeta(tarjeta: Tarjeta):
    """Create new tarjeta entries in the database"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            inserted_rows = []
            for row in tarjeta.rows:
                cursor.execute("""
                    INSERT INTO pagos
                    (id_documento, tipo_documento, ordenante)
                    VALUES (?, ?, ?)
                    RETURNING id_documento, tipo_documento, ordenante
                """, (
                    row.DOCUMENTO,
                    row.TIPO,
                    row.ORDENANTE
                ))
                
                # Get the inserted row data directly from RETURNING clause
                db_row = cursor.fetchone()
                if db_row:
                    inserted_rows.append(TarjetaRow(
                        DOCUMENTO=db_row[0],
                        TIPO=db_row[1],
                        ORDENANTE=db_row[3],
                    ))
            
            conn.commit()
            
            return Tarjeta(rows=inserted_rows)
            
    except Exception as e:
        logger.error(f"Error creating tarjeta entries: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tarjetas/", response_model=List[Tarjeta])
async def get_tarjetas(tipo_documento: Optional[str] = None):
    """Get all tarjeta entries, optionally filtered by tipo_documento"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Only get entries of type 'Tarjeta de credito' or 'Extracto movimiento'
            if tipo_documento:
                cursor.execute("""
                    SELECT id_documento, tipo_documento, ordenante, fecha_creacion
                    FROM pagos 
                    WHERE tipo_documento = ? 
                    AND tipo_documento IN ('Tarjeta de credito', 'Extracto movimiento', 'Arqueo de caja')
                    ORDER BY fecha_creacion DESC
                """, (tipo_documento,))
            else:
                cursor.execute("""
                    SELECT id_documento, tipo_documento, ordenante, fecha_creacion
                    FROM pagos
                    WHERE tipo_documento IN ('Tarjeta de credito', 'Extracto movimiento', 'Arqueo de caja')
                    ORDER BY fecha_creacion DESC
                """)
            
            # Group results by date to create Tarjeta objects with multiple rows
            results = {}
            for row in cursor.fetchall():
                results.append(TarjetaRow(
                    DOCUMENTO=row[0],
                    TIPO_DOCUMENTO=row[1],
                    ORDENANTE=row[2]
                ))
            
            # Convert grouped results to list of Tarjeta objects
            return results
            
    except Exception as e:
        logger.error(f"Error fetching tarjetas: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pagos_tabla/")
async def get_pagos_tabla():
    """Get all entries from pagos_tabla"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    concepto,
                    fecha_valor,
                    importe,
                    id_documento
                FROM pagos_tabla
            """)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "CONCEPTO": row[0],
                    "FECHA_VALOR": row[1],
                    "IMPORTE": row[2],
                    "ID_DOCUMENTO": row[3]
                })
            return results
            
    except Exception as e:
        logger.error(f"Error fetching pagos tabla: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/facturas/", response_model=FacturaRow)
async def create_factura(factura: FacturaRow):
    """Create a new factura in the database"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO facturas
                (cif_cliente, cliente, id_documento, numero_factura, fecha_factura, 
                proveedor, base_imponible, cif_proveedor, irpf, iva, total_factura)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                factura.CIF_CLIENTE,
                factura.CLIENTE,
                factura.FICHERO,
                factura.NUMERO_FACTURA,
                factura.FECHA_FACTURA,
                factura.PROVEEDOR,
                factura.BASE_IMPONIBLE,
                factura.CIF_PROVEEDOR,
                factura.IRPF,
                factura.IVA,
                factura.TOTAL_FACTURA
            ))
            
            inserted_id = cursor.lastrowid
            conn.commit()
            
            # Fetch inserted record
            cursor.execute("""
                SELECT cif_cliente, cliente, id_documento, numero_factura, fecha_factura, 
                       proveedor, base_imponible, cif_proveedor, irpf, iva, total_factura,
                       id_documento as fichero
                FROM facturas 
                WHERE id = ?
            """, (inserted_id,))
            
            row = cursor.fetchone()
            return FacturaRow(
                CIF_CLIENTE=row[0],
                CLIENTE=row[1],
                FICHERO=row[2],
                NUMERO_FACTURA=row[3],
                FECHA_FACTURA=row[4],
                PROVEEDOR=row[5],
                BASE_IMPONIBLE=row[6],
                CIF_PROVEEDOR=row[7],
                IRPF=row[8],
                IVA=row[9],
                TOTAL_FACTURA=row[10]
            )
            
    except Exception as e:
        logger.error(f"Error creating factura: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/facturas/", response_model=List[FacturaRow])
async def get_facturas():
    """Get all factura entries"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT cif_cliente, cliente, id_documento, numero_factura, fecha_factura, 
                       proveedor, base_imponible, cif_proveedor, irpf, iva, total_factura,
                       id_documento as fichero
                FROM facturas
                ORDER BY fecha_factura, fecha_creacion DESC
            """)
            
            results = []
            for row in cursor.fetchall():
                results.append(FacturaRow(
                    CIF_CLIENTE=row[0],
                    CLIENTE=row[1],
                    FICHERO=row[2],
                    NUMERO_FACTURA=row[3],
                    FECHA_FACTURA=row[4],
                    PROVEEDOR=row[5],
                    BASE_IMPONIBLE=row[6],
                    CIF_PROVEEDOR=row[7],
                    IRPF=row[8],
                    IVA=row[9],
                    TOTAL_FACTURA=row[10]
                ))
            
            return results
            
    except Exception as e:
        logger.error(f"Error fetching facturas: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/facturas_tabla/")
async def get_facturas_tabla():
    """Get all entries from facturas_tabla"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    concepto,
                    unidades,
                    importe,
                    id_documento
                FROM facturas_tabla
            """)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "CONCEPTO": row[0],
                    "UNIDADES": row[1],
                    "IMPORTE": row[2],
                    "ID_DOCUMENTO": row[3]
                })
            return results
            
    except Exception as e:
        logger.error(f"Error fetching facturas tabla: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/nominas/", response_model=NominaRow)
async def create_nomina(nomina: NominaRow):
    """Create a new nomina in the database"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO nominas
                (id_documento, mes, fecha_inicio, fecha_fin, cif, trabajador, naf, nif, categoria, antiguedad, 
                contrato, total_devengos, total_deducciones, absentismos, bc_teorica, prorrata, bc_con_complementos, 
                total_seg_social, bonificaciones_ss_trabajador, total_retenciones, total_retenciones_ss, 
                liquido_a_percibir, a_abonar, total_cuota_empresarial)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                nomina.ID_DOCUMENTO,
                nomina.MES,
                nomina.FECHA_INICIO,
                nomina.FECHA_FIN,
                nomina.CIF,
                nomina.TRABAJADOR,
                nomina.NAF,
                nomina.NIF,
                nomina.CATEGORIA,
                nomina.ANTIGUEDAD,
                nomina.CONTRATO,
                nomina.TOTAL_DEVENGOS,
                nomina.TOTAL_DEDUCCIONES,
                nomina.ABSENTISMOS,
                nomina.BC_TEORICA,
                nomina.PRORRATA,
                nomina.BC_CON_COMPLEMENTOS,
                nomina.TOTAL_SEG_SOCIAL,
                nomina.BONIFICACIONES_SS_TRABAJADOR,
                nomina.TOTAL_RETENCIONES,
                nomina.TOTAL_RETENCIONES_SS,
                nomina.LIQUIDO_A_PERCIBIR,
                nomina.A_ABONAR,
                nomina.TOTAL_CUOTA_EMPRESARIAL
            ))
            
            inserted_id = cursor.lastrowid
            conn.commit()
            
            # Fetch inserted record
            cursor.execute("""
                SELECT id_documento, cif, trabajador, mes, fecha_inicio, fecha_fin, 
                       naf, nif, categoria, antiguedad, contrato, total_devengos, total_deducciones, absentismos, bc_teorica, prorrata, bc_con_complementos, total_seg_social, bonificaciones_ss_trabajador, total_retenciones, total_retenciones_ss, liquido_a_percibir, a_abonar, total_cuota_empresarial
                FROM nominas 
                WHERE id = ?
            """, (inserted_id,))
            
            row = cursor.fetchone()
            return NominaRow(
                ID_DOCUMENTO=row[0],
                CIF=row[1],
                TRABAJADOR=row[2],
                MES=row[3],
                FECHA_INICIO=row[4],
                FECHA_FIN=row[5],
                NAF=row[6],
                NIF=row[7],
                CATEGORIA=row[8],
                ANTIGUEDAD=row[9],
                CONTRATO=row[10],
                TOTAL_DEVENGOS=row[11],
                TOTAL_DEDUCCIONES=row[12],
                ABSENTISMOS=row[13],
                BC_TEORICA=row[14],
                PRORRATA=row[15],
                BC_CON_COMPLEMENTOS=row[16],
                TOTAL_SEG_SOCIAL=row[17],
                BONIFICACIONES_SS_TRABAJADOR=row[18],
                TOTAL_RETENCIONES=row[19],
                TOTAL_RETENCIONES_SS=row[20],
                LIQUIDO_A_PERCIBIR=row[21],
                A_ABONAR=row[22],
                TOTAL_CUOTA_EMPRESARIAL=row[23]
            )
            
    except Exception as e:
        logger.error(f"Error creating factura: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/nominas/", response_model=List[NominaRow])
async def get_nominas():
    """Get all nomina entries"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id_documento, cif, trabajador, mes, fecha_inicio, fecha_fin, 
                       naf, nif, categoria, antiguedad, contrato, total_devengos, total_deducciones, absentismos, bc_teorica, prorrata, bc_con_complementos, total_seg_social, bonificaciones_ss_trabajador, total_retenciones, total_retenciones_ss, liquido_a_percibir, a_abonar, total_cuota_empresarial
                FROM nominas
                ORDER BY fecha_inicio, fecha_creacion DESC
            """)
            
            results = []
            for row in cursor.fetchall():
                results.append(NominaRow(
                    ID_DOCUMENTO=row[0],
                    CIF=row[1],
                    TRABAJADOR=row[2],
                    MES=row[3],
                    FECHA_INICIO=row[4],
                    FECHA_FIN=row[5],
                    NAF=row[6],
                    NIF=row[7],
                    CATEGORIA=row[8],
                    ANTIGUEDAD=row[9],
                    CONTRATO=row[10],
                    TOTAL_DEVENGOS=row[11],
                    TOTAL_DEDUCCIONES=row[12],
                    ABSENTISMOS=row[13],
                    BC_TEORICA=row[14],
                    PRORRATA=row[15],
                    BC_CON_COMPLEMENTOS=row[16],
                    TOTAL_SEG_SOCIAL=row[17],
                    BONIFICACIONES_SS_TRABAJADOR=row[18],
                    TOTAL_RETENCIONES=row[19],
                    TOTAL_RETENCIONES_SS=row[20],
                    LIQUIDO_A_PERCIBIR=row[21],
                    A_ABONAR=row[22],
                    TOTAL_CUOTA_EMPRESARIAL=row[23]
                ))
            
            return results
            
    except Exception as e:
        logger.error(f"Error fetching facturas: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
  
@app.get("/nominas_tabla/")
async def get_nominas_tabla():
    """Get all entries from nominas_tabla"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    descripcion,
                    importe_unidad,
                    unidad,
                    devengos,
                    deducciones
                FROM nominas_tabla
            """)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "DESCRIPCION": row[0],
                    "IMPORTE_UNIDAD": row[1],
                    "UNIDAD": row[2],
                    "DEVENGOS": row[3],
                    "DEDUCCIONES": row[4]
                })
            return results
            
    except Exception as e:
        logger.error(f"Error fetching nominas tabla: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images/facturas", response_model=List[ImageResponse])
async def list_images(processed: Optional[bool] = None):
    """
    List all images in the system.
    If processed=True, only show processed images.
    If processed=False, only show unprocessed images.
    If processed=None, show all images.
    """
    try:
        images_dir = IMAGES_DIR / "facturas"
        processed_dir = Path(os.getenv("PROCESSED_IMAGES_DIR", "./Processed"))
        images_dir.mkdir(exist_ok=True)
        processed_dir.mkdir(exist_ok=True)

        images = []
        
        def scan_directory(directory: Path) -> List[Dict]:
            results = []
            for file_path in directory.glob("*"):
                if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    file_stat = file_path.stat()
                    results.append(ImageResponse(
                        filename=file_path.name,
                        path=str(file_path),
                        size=file_stat.st_size,
                        last_modified=datetime.fromtimestamp(file_stat.st_mtime)
                    ))
            return results

        if processed is None:
            # Get all images
            images.extend(scan_directory(images_dir))
            images.extend(scan_directory(processed_dir))
        elif processed:
            # Get only processed images
            images.extend(scan_directory(processed_dir))
        else:
            # Get only unprocessed images
            images.extend(scan_directory(images_dir))

        return images

    except Exception as e:
        logger.error(f"Error listing images: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images/pagos", response_model=List[ImageResponse])
async def list_images(processed: Optional[bool] = None):
    """
    List all images in the system.
    If processed=True, only show processed images.
    If processed=False, only show unprocessed images.
    If processed=None, show all images.
    """
    try:
        images_dir = images_dir = IMAGES_DIR / "pagos"
        processed_dir = Path(os.getenv("PROCESSED_IMAGES_DIR", "./Processed"))
        images_dir.mkdir(exist_ok=True)
        processed_dir.mkdir(exist_ok=True)

        images = []
        
        def scan_directory(directory: Path) -> List[Dict]:
            results = []
            for file_path in directory.glob("*"):
                if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    file_stat = file_path.stat()
                    results.append(ImageResponse(
                        filename=file_path.name,
                        path=str(file_path),
                        size=file_stat.st_size,
                        last_modified=datetime.fromtimestamp(file_stat.st_mtime)
                    ))
            return results

        if processed is None:
            # Get all images
            images.extend(scan_directory(images_dir))
            images.extend(scan_directory(processed_dir))
        elif processed:
            # Get only processed images
            images.extend(scan_directory(processed_dir))
        else:
            # Get only unprocessed images
            images.extend(scan_directory(images_dir))

        return images

    except Exception as e:
        logger.error(f"Error listing images: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images/nominas", response_model=List[ImageResponse])
async def list_images(processed: Optional[bool] = None):
    """
    List all images in the system.
    If processed=True, only show processed images.
    If processed=False, only show unprocessed images.
    If processed=None, show all images.
    """
    try:
        images_dir = IMAGES_DIR / "nominas"
        processed_dir = Path(os.getenv("PROCESSED_IMAGES_DIR", "./Processed"))
        images_dir.mkdir(exist_ok=True)
        processed_dir.mkdir(exist_ok=True)

        images = []
        
        def scan_directory(directory: Path) -> List[Dict]:
            results = []
            for file_path in directory.glob("*"):
                if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    file_stat = file_path.stat()
                    results.append(ImageResponse(
                        filename=file_path.name,
                        path=str(file_path),
                        size=file_stat.st_size,
                        last_modified=datetime.fromtimestamp(file_stat.st_mtime)
                    ))
            return results

        if processed is None:
            # Get all images
            images.extend(scan_directory(images_dir))
            images.extend(scan_directory(processed_dir))
        elif processed:
            # Get only processed images
            images.extend(scan_directory(processed_dir))
        else:
            # Get only unprocessed images
            images.extend(scan_directory(images_dir))

        return images

    except Exception as e:
        logger.error(f"Error listing images: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/images/upload/{doc_type}")
async def upload_files(
    doc_type: Literal["facturas", "pagos", "nominas"],
    file: UploadFile = File(...)
) -> List[ImageResponse]:
    """Upload and process documents"""
    try:
        # Read file content
        content = await file.read()
        filename = file.filename
        
        # Determine file type and validate
        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type:
            mime_type = file.content_type
            
        if mime_type == 'application/pdf':
            # Convert PDF to images
            images = convert_from_bytes(content)
            results = []
            
            for idx, image in enumerate(images):
                # Generate unique filename for each page
                base_name = Path(filename).stem
                image_filename = f"{base_name}_page{idx+1}.jpg"
                
                # Convert image to bytes
                img_byte_arr = BytesIO()
                image.save(img_byte_arr, format='JPEG')
                image_content = img_byte_arr.getvalue()
                
                # Save to filesystem and database
                result = await upload_images_by_type(
                    image_content=image_content,
                    filename=image_filename,
                    doc_type=doc_type
                )
                results.append(result)
                
            return results
            
        elif mime_type and mime_type.startswith('image/'):
            # Handle single image upload
            result = await upload_images_by_type(
                image_content=content,
                filename=filename,
                doc_type=doc_type
            )
            return [result]
            
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload PDF or image files."
            )
            
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def upload_images_by_type(image_content: bytes, filename: str, doc_type: str) -> ImageResponse:
    """Save image to specific document type directory and database table"""
    try:
        # Save to filesystem
        dest_dir = IMAGES_DIR / doc_type
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename
        
        async with aiofiles.open(dest_path, "wb") as f:
            await f.write(image_content)
        
        # Determine mime type
        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type:
            mime_type = 'image/jpeg'  # default to jpeg if can't determine
        
        # Save to database
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                INSERT INTO images_{doc_type} 
                (filename, content, size, mime_type)
                VALUES (?, ?, ?, ?)
            """, (
                filename,
                image_content,
                len(image_content),
                mime_type
            ))
            conn.commit()
        
        # Create response
        file_stat = dest_path.stat()
        return ImageResponse(
            filename=filename,
            path=str(dest_path),
            size=file_stat.st_size,
            last_modified=datetime.fromtimestamp(file_stat.st_mtime)
        )
        
    except Exception as e:
        logger.error(f"Error in upload_images_by_type: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save image: {str(e)}"
        )

async def store_processed_image(original_filename: str, processed_content: bytes):
    """Store a processed image in the database and filesystem"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Get original image ID
            cursor.execute("SELECT id FROM images WHERE filename = ?", (original_filename,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Original image {original_filename} not found")
            
            original_id = row[0]
            processed_filename = f"processed_{original_filename}"
            
            # Save to processed_images table
            cursor.execute("""
                INSERT INTO processed_images 
                (original_image_id, filename, content, size)
                VALUES (?, ?, ?, ?)
            """, (
                original_id,
                processed_filename,
                processed_content,
                len(processed_content)
            ))
            conn.commit()
            
            # Save to filesystem
            processed_dir = Path(os.getenv("PROCESSED_IMAGES_DIR", "./Processed"))
            processed_dir.mkdir(exist_ok=True)
            processed_path = processed_dir / processed_filename
            
            with open(processed_path, "wb") as f:
                f.write(processed_content)
                
            return processed_filename
            
    except Exception as e:
        logger.error(f"Error storing processed image: {str(e)}")
        raise

@app.get("/images/{filename}/download")
async def download_image(filename: str):
    """Download an image from the database"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Try to get from processed images first
            cursor.execute("""
                SELECT images.content, images.mime_type 
                FROM processed_images 
                JOIN images ON processed_images.original_image_id = images.id
                WHERE processed_images.filename = ?
            """, (filename,))
            
            row = cursor.fetchone()
            
            # If not found in processed, try original images
            if not row:
                cursor.execute("""
                    SELECT content, mime_type 
                    FROM images 
                    WHERE filename = ?
                """, (filename,))
                row = cursor.fetchone()
            
            if not row:
                raise HTTPException(
                    status_code=404,
                    detail=f"Image {filename} not found"
                )
            
            content, mime_type = row
            
            return Response(
                content=content,
                media_type=mime_type,
                headers={
                    "Content-Disposition": f"attachment; filename={filename}"
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/images/{filename}/process/facturas")
async def process_facturas(filename: str, rewrite: bool = False):
    """
    Process images through the facturas workflow.
    """
    try:
        # Extract base filename without extension and number
        base_name = Path(filename).stem
        base_name = re.sub(r'\(\d+\)$', '', base_name)  # Remove (n) if present
        file_ext = Path(filename).suffix

        # Find all matching files
        matching_files = []
        images_facturas_dir = IMAGES_DIR / "facturas"
        for file in images_facturas_dir.glob(f"{base_name}*{file_ext}"):
            if file.exists():
                matching_files.append(file.name)

        if not matching_files:
            raise HTTPException(
                status_code=404,
                detail=f"No images found matching pattern {base_name}*{file_ext}"
            )

        # Process through facturas workflow
        workflow_manager = WorkflowManager()
        result = await workflow_manager.process_workflow(
            workflow_type="facturas",
            image_paths=matching_files,
            image_directory=str(images_facturas_dir)
        )

        return {
            "message": f"Processed through facturas workflow",
            "base_pattern": f"{base_name}*{file_ext}",
            "result": result,
            "data_stored": result.get("status") == "success"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing facturas: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/images/{filename}/process/pagos")
async def process_pagos(filename: str, rewrite: bool = False):
    """
    Process images through the pagos workflow.
    """
    try:
        # Extract base filename without extension and number
        base_name = Path(filename).stem
        base_name = re.sub(r'\(\d+\)$', '', base_name)  # Remove (n) if present
        file_ext = Path(filename).suffix

        # Find all matching files
        matching_files = []
        images_pagos_dir = IMAGES_DIR / "pagos"
        for file in images_pagos_dir.glob(f"{base_name}*{file_ext}"):
            if file.exists():
                matching_files.append(file.name)

        if not matching_files:
            raise HTTPException(
                status_code=404,
                detail=f"No images found matching pattern {base_name}*{file_ext}"
            )

        # Process through pagos workflow
        workflow_manager = WorkflowManager()
        result = await workflow_manager.process_workflow(
            workflow_type="pagos",
            image_paths=matching_files,
            image_directory=str(images_pagos_dir)
        )

        return {
            "message": f"Processed through pagos workflow",
            "base_pattern": f"{base_name}*{file_ext}",
            "result": result,
            "data_stored": result.get("status") == "success"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing pagos: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/images/{filename}/process/nominas")
async def process_nominas(filename: str, rewrite: bool = False):
    """
    Process images through the nominas workflow.
    """
    try:
        # Extract base filename without extension and number
        base_name = Path(filename).stem
        base_name = re.sub(r'\(\d+\)$', '', base_name)  # Remove (n) if present
        file_ext = Path(filename).suffix

        # Find all matching files
        matching_files = []
        images_nominas_dir = IMAGES_DIR / "nominas"
        for file in images_nominas_dir.glob(f"{base_name}*{file_ext}"):
            if file.exists():
                matching_files.append(file.name)

        if not matching_files:
            raise HTTPException(
                status_code=404,
                detail=f"No images found matching pattern {base_name}*{file_ext}"
            )

        # Process through facturas workflow
        workflow_manager = WorkflowManager()
        result = await workflow_manager.process_workflow(
            workflow_type="nominas",
            image_paths=matching_files,
            image_directory=str(images_nominas_dir)
        )

        return {
            "message": f"Processed through nominas workflow",
            "base_pattern": f"{base_name}*{file_ext}",
            "result": result,
            "data_stored": result.get("status") == "success"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing nominas: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Update the original process endpoint to be deprecated
@app.post("/images/{filename}/process")
async def process_image(filename: str, rewrite: bool = False):
    """
    [DEPRECATED] Use /process/facturas or /process/pagos or /process/nominas instead.
    Process images through all available workflows concurrently.
    """
    logger.warning("This endpoint is deprecated. Use /process/facturas or /process/pagos or /process/nominas instead.")
    try:
        # Extract base filename without extension and number
        base_name = Path(filename).stem
        base_name = re.sub(r'\(\d+\)$', '', base_name)  # Remove (n) if present
        file_ext = Path(filename).suffix

        # Find all matching files
        matching_files = []
        for file in IMAGES_DIR.glob(f"{base_name}*{file_ext}"):
            if file.exists():
                matching_files.append(file.name)

        if not matching_files:
            raise HTTPException(
                status_code=404,
                detail=f"No images found matching pattern {base_name}*{file_ext}"
            )

        # Process through all workflows concurrently
        workflow_manager = WorkflowManager()
        results = await workflow_manager.process_all_workflows(
            image_paths=matching_files,
            image_directory=str(IMAGES_DIR)
        )

        # Summarize results
        total_files = len(matching_files)
        successful_workflows = sum(1 for r in results.values() if r and r["status"] == "success")

        return {
            "message": f"[DEPRECATED] Processed through {successful_workflows} workflows. Use /process/facturas or /process/pagos or /process/nominas instead.",
            "base_pattern": f"{base_name}*{file_ext}",
            "results": results,
            "data_stored": True
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing images: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add this synchronous version of process_single_image
def process_single_image_sync(source_path: str):
    """
    Synchronous wrapper for process_single_image to use with ProcessPoolExecutor
    """
    async def run():
        return await process_single_image(source_path)
        
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(run())
    finally:
        loop.close()

@app.delete("/images/delete")
async def delete_all_images(delete_transactions: bool = False):
    """
    Delete all images and related data from the system.
    
    Args:
        delete_transactions: If True, also deletes all transactions from pagos table
    """
    try:
        # Delete files from filesystem
        images_facturas_dir = IMAGES_DIR / "facturas"
        images_pagos_dir = IMAGES_DIR / "pagos"
        images_nominas_dir = IMAGES_DIR / "nominas"
        processed_dir = Path(os.getenv("PROCESSED_IMAGES_DIR", "./Processed"))
        excel_dir = Path(os.getenv("EXCEL_OUTPUT_DIR", "./output"))
        
        # Function to safely delete files in a directory with specific extensions
        def clean_directory(directory: Path, extensions: list = None):
            if directory.exists():
                for file in directory.glob("*"):
                    if file.is_file():
                        if extensions is None or file.suffix.lower() in extensions:
                            try:
                                file.unlink()
                                logger.info(f"Deleted file: {file}")
                            except Exception as e:
                                logger.error(f"Error deleting file {file}: {str(e)}")

        # Clean directories with specific extensions
        clean_directory(images_facturas_dir, ['.png', '.jpg', '.jpeg'])
        clean_directory(images_pagos_dir, ['.png', '.jpg', '.jpeg'])
        clean_directory(images_nominas_dir, ['.png', '.jpg', '.jpeg'])
        clean_directory(processed_dir, ['.png', '.jpg', '.jpeg'])
        clean_directory(excel_dir, ['.xlsx', '.xls'])
        
        deleted_counts = {
            "filesystem": {
                "images_directory": "cleaned",
                "processed_directory": "cleaned",
                "temp_directory": "cleaned",
                "excel_directory": "cleaned"
            }
        }
        
        # Delete records from database
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Delete from processed_images first (due to foreign key constraint)
            cursor.execute("DELETE FROM processed_images")
            deleted_counts["processed_images"] = cursor.rowcount
            
            # Then delete from images
            cursor.execute("DELETE FROM images_pagos")
            deleted_counts["original_images"] = cursor.rowcount
            
            cursor.execute("DELETE FROM images_facturas")
            deleted_counts["original_images"] = cursor.rowcount
            
            cursor.execute("DELETE FROM images_nominas")
            deleted_counts["original_images"] = cursor.rowcount
            
            # Optionally delete from pagos
            if delete_transactions:
                cursor.execute("DELETE FROM facturas")
                deleted_counts["transactions"] = cursor.rowcount
                
                cursor.execute("DELETE FROM facturas_tabla")
                deleted_counts["transactions"] = cursor.rowcount
                
                cursor.execute("DELETE FROM pagos_tabla")
                deleted_counts["transactions"] = cursor.rowcount
                
                cursor.execute("DELETE FROM pagos")
                deleted_counts["transactions"] = cursor.rowcount
                
                cursor.execute("DELETE FROM nominas")
                deleted_counts["transactions"] = cursor.rowcount
                
                cursor.execute("DELETE FROM nominas_tabla")
                deleted_counts["transactions"] = cursor.rowcount
                
            conn.commit()
            
        return {
            "message": "Successfully deleted all data",
            "deleted_counts": deleted_counts
        }
        
    except Exception as e:
        logger.error(f"Error deleting data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workflow/graph")
async def get_workflow_graph():
    """
    Get the workflow graph in Mermaid format.
    Returns both the Mermaid string and a rendered PNG (if available).
    """
    try:
        # Instatiate a workflowmanager
        workflowmanager = WorkflowManager()
        
        # Get the workflow graph
        workflow_graph = workflowmanager.get_graph()
        
        # Generate PNG file
        output_path = "Agentes.png"
        
        # Save the graph as PNG
        workflow_graph.draw_mermaid_png(output_file_path=output_path)
        
        # Check if file was created successfully
        if output_path.exists():
            return {
                "success": True,
                "message": "Graph exported successfully",
                "file_path": str(output_path)
            }
        else:
            return {
                "success": False, 
                "message": "Failed to generate graph image",
                "file_path": None
            }
            
    except (ImportError, Exception) as e:
        logger.warning(f"Could not generate PNG: {str(e)}")
        return {
            "success": False,
            "message": str(e),
            "file_path": None
        }
            
    except Exception as e:
        logger.error(f"Error generating workflow graph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/excel/download")
async def download_excel():
    """Download an Excel file containing all four tables in separate sheets"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Create DataFrames for tablas
            facturas_tabla_df = pd.read_sql_query("""
                SELECT 
                    concepto as 'Concepto',
                    unidades as 'Unidades',
                    importe as 'Importe',
                    id_documento as 'ID Documento'
                FROM facturas_tabla
            """, conn)
            
            pagos_tabla_df = pd.read_sql_query("""
                SELECT 
                    concepto as 'Concepto',
                    fecha_valor as 'Fecha Valor',
                    importe as 'Importe',
                    id_documento as 'ID Documento'
                FROM pagos_tabla
            """, conn)
            
            # Create DataFrames for main tables
            facturas_df = pd.read_sql_query("""
                SELECT 
                    cif_cliente as 'CIF Cliente',
                    cliente as 'Cliente',
                    id_documento as 'ID Documento',
                    numero_factura as 'Número Factura',
                    fecha_factura as 'Fecha Factura',
                    proveedor as 'Proveedor',
                    base_imponible as 'Base Imponible',
                    cif_proveedor as 'CIF Proveedor',
                    irpf as 'IRPF',
                    iva as 'IVA',
                    total_factura as 'Total Factura'
                FROM facturas
            """, conn)
            
            pagos_df = pd.read_sql_query("""
                SELECT 
                    id_documento as 'ID Documento',
                    tipo_documento as 'Tipo',
                    fecha_valor as 'Fecha Valor',
                    ordenante as 'Ordenante',
                    beneficiario as 'Beneficiario',
                    concepto as 'Concepto',
                    importe as 'Importe'
                FROM pagos
            """, conn)
            
            nominas_df = pd.read_sql_query("""
                SELECT 
                    id_documento as 'ID Documento',
                    mes as 'Mes',
                    fecha_inicio as 'Fecha Inicio',
                    fecha_fin as 'Fecha Fin',
                    cif as 'CIF',
                    trabajador as 'Trabajador',
                    naf as 'NAF',
                    nif as 'NIF',
                    categoria as 'Categoria',
                    antiguedad as 'Antiguedad',
                    contrato as 'Contrato',
                    total_devengos as 'Total Devengos',
                    total_deducciones as 'Total Deducciones',
                    absentismos as 'Absentismos',
                    bc_teorica as 'BC Teorica',
                    prorrata as 'Prorrata',
                    bc_con_complementos as 'BC Con Complementos',
                    total_seg_social as 'Total Seg Social',
                    bonificaciones_ss_trabajador as 'Bonificaciones SS Trabajador',
                    total_retenciones as 'Total Retenciones',
                    total_retenciones_ss as 'Total Retenciones SS',
                    liquido_a_percibir as 'Liquido a Percibir',
                    a_abonar as 'A Abonar',
                    total_cuota_empresarial as 'Total Cuota Empresarial'
                FROM nominas
            """, conn)
            
            tabla_nominas_df = pd.read_sql_query("""
                SELECT 
                    id_documento as 'ID Documento',
                    descripcion as 'Descripción',
                    importe_unidad as 'Importe Unidad',
                    unidad as 'Unidad',
                    devengos as 'Devengos',
                    deducciones as 'Deducciones'
                FROM nominas_tabla
            """, conn)
            

            # Create Excel file in memory with all sheets
            excel_file = BytesIO()
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                if not facturas_df.empty:
                    facturas_df.to_excel(writer, sheet_name='Facturas', index=False)
                    facturas_tabla_df.to_excel(writer, sheet_name='Facturas Desglose', index=False)
                if not pagos_df.empty:
                    pagos_df.to_excel(writer, sheet_name='Pagos', index=False)
                    pagos_tabla_df.to_excel(writer, sheet_name='Pagos Desglose', index=False)
                if not nominas_df.empty:
                    nominas_df.to_excel(writer, sheet_name='Nominas', index=False)
                    tabla_nominas_df.to_excel(writer, sheet_name='Nominas Desglose', index=False)

                # Auto-adjust columns width for all sheets
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    for idx, col in enumerate(worksheet.columns, 1):
                        max_length = 0
                        column = worksheet.column_dimensions[chr(64 + idx)]
                        
                        for cell in col:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        
                        adjusted_width = (max_length + 2)
                        column.width = min(adjusted_width, 50)
            
            excel_file.seek(0)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"autoaudit_{timestamp}.xlsx"
            
            return Response(
                content=excel_file.getvalue(),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
            
    except Exception as e:
        logger.error(f"Error generating Excel file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add this utility function
async def store_data(result: dict) -> bool:
    """Store processed data with proper error handling"""
    try:
        if result["data_type"] == "tarjeta":
            await create_tarjeta(result["data"])
            return True
        elif result["data_type"] == "transferencia":
            await create_transferencia(result["data"])
            return True
        elif result["data_type"] == "factura":
            await create_factura(result["data"])
            return True
        return False
    except Exception as e:
        logger.error(f"Error storing data: {str(e)}")
        return False

# Cache expensive computations
# @lru_cache(maxsize=128)
# async def expensive_computation(data: str):
#     # Your computation here
#     pass

# # Use background tasks for non-critical operations
# @app.post("/process")
# async def process_data(data: dict, background_tasks: BackgroundTasks):
#     # Handle immediate response
#     result = await process_immediate(data)
    
#     # Queue non-critical tasks
#     background_tasks.add_task(process_background, data)
    
#     return result

@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Read PDF content
        pdf_content = await file.read()
        
        # Convert PDF to images
        images = convert_from_bytes(pdf_content)
        
        # Process each image
        results = []
        for image in images:
            # Convert image to bytes
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
            
            # Process image (e.g., send to Textract or VLM)
            # result = process_image(img_bytes)
            # results.append(result)
        
        return {"message": "PDF processed successfully", "results": results}
    
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("flow.backend.api:app", host="0.0.0.0", 
                port=int(os.getenv("SERVER_PORT", "8000")), 
                reload=True) 