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
BASE_RATE_LIMIT = 50000  # requests per minute
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
                    comments TEXT,
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

@app.get("/health")
def health_check():
    return JSONResponse(content={"status": "ok"})

@app.on_event("startup")
async def startup_event():
    """Initialize database and process pool on startup with retry logic"""
    global process_pool
    init_db()
    
    for attempt in range(MAX_RETRIES):
        try:
            process_pool = ProcessPoolExecutor(max_workers=MAX_WORKERS)
            break
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
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
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/nominas/", response_model=NominaRow)
async def create_nomina(nomina: NominaRow):
    """Create a new nomina in the database"""
    try:
        # Check if CIF was corrected by validate_and_extract_cif
        original_cif = nomina.CIF
        
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO nominas
                (id_documento, mes, fecha_inicio, fecha_fin, cif, trabajador, naf, nif, categoria, antiguedad, 
                contrato, total_devengos, total_deducciones, absentismos, bc_teorica, prorrata, bc_con_complementos, 
                total_seg_social, bonificaciones_ss_trabajador, total_retenciones, total_retenciones_ss, 
                liquido_a_percibir, a_abonar, total_cuota_empresarial, comments)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                nomina.TOTAL_CUOTA_EMPRESARIAL,
                nomina.COMMENTS
            ))
            
            inserted_id = cursor.lastrowid
            conn.commit()
            
            # Fetch inserted record
            cursor.execute("""
                SELECT id_documento, cif, trabajador, mes, fecha_inicio, fecha_fin, 
                       naf, nif, categoria, antiguedad, contrato, total_devengos, total_deducciones, absentismos, 
                       bc_teorica, prorrata, bc_con_complementos, total_seg_social, bonificaciones_ss_trabajador, 
                       total_retenciones, total_retenciones_ss, liquido_a_percibir, a_abonar, total_cuota_empresarial, comments
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
                TOTAL_CUOTA_EMPRESARIAL=row[23],
                COMMENTS=row[24]
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/nominas/", response_model=List[NominaRow])
async def get_nominas():
    """Get all nomina entries"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id_documento, cif, trabajador, mes, fecha_inicio, fecha_fin, 
                       naf, nif, categoria, antiguedad, contrato, total_devengos, total_deducciones, absentismos, 
                       bc_teorica, prorrata, bc_con_complementos, total_seg_social, bonificaciones_ss_trabajador, 
                       total_retenciones, total_retenciones_ss, liquido_a_percibir, a_abonar, total_cuota_empresarial, comments
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
                    TOTAL_CUOTA_EMPRESARIAL=row[23],
                    COMMENTS=row[24]
                ))
            
            return results
            
    except Exception as e:
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
                    devengos,
                    deducciones
                FROM nominas_tabla
            """)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "DESCRIPCION": row[0],
                    "DEVENGOS": row[1],
                    "DEDUCCIONES": row[2]
                })
            return results
            
    except Exception as e:
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

        # If no images found in filesystem, try to retrieve from database
        if not images:
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    cursor = conn.cursor()
                    
                    # Query database for image metadata
                    query = "SELECT filename, size, mime_type, upload_date FROM images_facturas"
                    
                    if processed is not None:
                        # In the future, add filtering by processed status if needed
                        pass
                        
                    cursor.execute(query)
                    
                    for row in cursor.fetchall():
                        filename, size, mime_type, upload_date = row
                        # Create virtual path since file doesn't exist on disk
                        virtual_path = str(images_dir / filename)
                        
                        # Parse timestamp
                        if isinstance(upload_date, str):
                            try:
                                last_modified = datetime.fromisoformat(upload_date.replace('Z', '+00:00'))
                            except (ValueError, TypeError):
                                last_modified = datetime.now()
                        else:
                            last_modified = datetime.now()
                            
                        images.append(ImageResponse(
                            filename=filename,
                            path=virtual_path,
                            size=size,
                            last_modified=last_modified
                        ))
                        
                    logger.info(f"Retrieved {len(images)} facturas images from database")
                    
                    # Optionally restore files to filesystem
                    if images and processed is False:  # Only for unprocessed images
                        for img in images[:10]:  # Limit to first 10 to avoid performance issues
                            try:
                                # Get file content from database
                                cursor.execute("SELECT content FROM images_facturas WHERE filename = ?", (img.filename,))
                                content_row = cursor.fetchone()
                                
                                if content_row and content_row[0]:
                                    # Save to filesystem
                                    img_path = images_dir / img.filename
                                    with open(img_path, 'wb') as f:
                                        f.write(content_row[0])
                                    logger.info(f"Restored file {img.filename} to filesystem")
                            except Exception as restore_err:
                                logger.warning(f"Error restoring file {img.filename}: {str(restore_err)}")
                
            except Exception as db_err:
                logger.error(f"Error retrieving images from database: {str(db_err)}")
                # Continue with empty images list if database retrieval fails

        return images

    except Exception as e:
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
        images_dir = IMAGES_DIR / "pagos"
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

        # If no images found in filesystem, try to retrieve from database
        if not images:
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    cursor = conn.cursor()
                    
                    # Query database for image metadata
                    query = "SELECT filename, size, mime_type, upload_date FROM images_pagos"
                    
                    if processed is not None:
                        # In the future, add filtering by processed status if needed
                        pass
                        
                    cursor.execute(query)
                    
                    for row in cursor.fetchall():
                        filename, size, mime_type, upload_date = row
                        # Create virtual path since file doesn't exist on disk
                        virtual_path = str(images_dir / filename)
                        
                        # Parse timestamp
                        if isinstance(upload_date, str):
                            try:
                                last_modified = datetime.fromisoformat(upload_date.replace('Z', '+00:00'))
                            except (ValueError, TypeError):
                                last_modified = datetime.now()
                        else:
                            last_modified = datetime.now()
                            
                        images.append(ImageResponse(
                            filename=filename,
                            path=virtual_path,
                            size=size,
                            last_modified=last_modified
                        ))
                        
                    logger.info(f"Retrieved {len(images)} pagos images from database")
                    
                    # Optionally restore files to filesystem
                    if images and processed is False:  # Only for unprocessed images
                        for img in images[:10]:  # Limit to first 10 to avoid performance issues
                            try:
                                # Get file content from database
                                cursor.execute("SELECT content FROM images_pagos WHERE filename = ?", (img.filename,))
                                content_row = cursor.fetchone()
                                
                                if content_row and content_row[0]:
                                    # Save to filesystem
                                    img_path = images_dir / img.filename
                                    with open(img_path, 'wb') as f:
                                        f.write(content_row[0])
                                    logger.info(f"Restored file {img.filename} to filesystem")
                            except Exception as restore_err:
                                logger.warning(f"Error restoring file {img.filename}: {str(restore_err)}")
                
            except Exception as db_err:
                logger.error(f"Error retrieving images from database: {str(db_err)}")
                # Continue with empty images list if database retrieval fails

        return images

    except Exception as e:
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

        # If no images found in filesystem, try to retrieve from database
        if not images:
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    cursor = conn.cursor()
                    
                    # Query database for image metadata
                    query = "SELECT filename, size, mime_type, upload_date FROM images_nominas"
                    
                    if processed is not None:
                        # In the future, add filtering by processed status if needed
                        pass
                        
                    cursor.execute(query)
                    
                    for row in cursor.fetchall():
                        filename, size, mime_type, upload_date = row
                        # Create virtual path since file doesn't exist on disk
                        virtual_path = str(images_dir / filename)
                        
                        # Parse timestamp
                        if isinstance(upload_date, str):
                            try:
                                last_modified = datetime.fromisoformat(upload_date.replace('Z', '+00:00'))
                            except (ValueError, TypeError):
                                last_modified = datetime.now()
                        else:
                            last_modified = datetime.now()
                            
                        images.append(ImageResponse(
                            filename=filename,
                            path=virtual_path,
                            size=size,
                            last_modified=last_modified
                        ))
                        
                    logger.info(f"Retrieved {len(images)} nominas images from database")
                    
                    # Optionally restore files to filesystem
                    if images and processed is False:  # Only for unprocessed images
                        for img in images[:10]:  # Limit to first 10 to avoid performance issues
                            try:
                                # Get file content from database
                                cursor.execute("SELECT content FROM images_nominas WHERE filename = ?", (img.filename,))
                                content_row = cursor.fetchone()
                                
                                if content_row and content_row[0]:
                                    # Save to filesystem
                                    img_path = images_dir / img.filename
                                    with open(img_path, 'wb') as f:
                                        f.write(content_row[0])
                                    logger.info(f"Restored file {img.filename} to filesystem")
                            except Exception as restore_err:
                                logger.warning(f"Error restoring file {img.filename}: {str(restore_err)}")
                
            except Exception as db_err:
                logger.error(f"Error retrieving images from database: {str(db_err)}")
                # Continue with empty images list if database retrieval fails

        return images

    except Exception as e:
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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/images/{filename}/process/nominas")
async def process_nominas(filename: str, rewrite: bool = False):
    """
    Process images through the nominas workflow.
    """
    try:
        # Extract base filename without extension and number
        base_name = Path(filename).stem
        # base_name = re.sub(r'\(\d+\)$', '', base_name)  # Remove (n) if present
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
        raise HTTPException(status_code=500, detail=str(e))

# Update the original process endpoint to be deprecated
@app.post("/images/{filename}/process")
async def process_image(filename: str, rewrite: bool = False):
    """
    [DEPRECATED] Use /process/facturas or /process/pagos or /process/nominas instead.
    Process images through all available workflows concurrently.
    """
    raise HTTPException(status_code=400, detail="This endpoint is deprecated. Use /process/facturas or /process/pagos or /process/nominas instead.")

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
async def delete_all_images(delete_transactions: bool = False, delete_pdf_history: bool = False):
    """
    Delete all images and related data from the system.
    
    Args:
        delete_transactions: If True, also deletes all transactions from pagos table
        delete_pdf_history: If True, also deletes all PDF conversion history
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
                            except Exception as e:
                                raise HTTPException(status_code=500, detail=f"Error deleting file {file}: {str(e)}")

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
            deleted_counts["original_images_pagos"] = cursor.rowcount
            
            cursor.execute("DELETE FROM images_facturas")
            deleted_counts["original_images_facturas"] = cursor.rowcount
            
            cursor.execute("DELETE FROM images_nominas")
            deleted_counts["original_images_nominas"] = cursor.rowcount
            
            # Optionally delete PDF conversion history
            if delete_pdf_history:
                cursor.execute("DELETE FROM pdf_conversion_history")
                deleted_counts["pdf_conversion_history"] = cursor.rowcount
            
            # Optionally delete from pagos
            if delete_transactions:
                cursor.execute("DELETE FROM facturas")
                deleted_counts["facturas"] = cursor.rowcount
                
                cursor.execute("DELETE FROM facturas_tabla")
                deleted_counts["facturas_tabla"] = cursor.rowcount
                
                cursor.execute("DELETE FROM pagos_tabla")
                deleted_counts["pagos_tabla"] = cursor.rowcount
                
                cursor.execute("DELETE FROM pagos")
                deleted_counts["pagos"] = cursor.rowcount
                
                cursor.execute("DELETE FROM nominas")
                deleted_counts["nominas"] = cursor.rowcount
                
                cursor.execute("DELETE FROM nominas_tabla")
                deleted_counts["nominas_tabla"] = cursor.rowcount
                
            conn.commit()
            
        return {
            "message": "Successfully deleted all data",
            "deleted_counts": deleted_counts
        }
        
    except Exception as e:
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
        raise HTTPException(status_code=500, detail=str(e))
            
    except Exception as e:
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
            
            # Create DataFrames for main tables - COMPLETELY EXCLUDE IRPF
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
                    iva as 'IVA',
                    total_factura as 'Total Factura'
                FROM facturas
            """, conn)
            
            # Add a placeholder IRPF column with zeros
            facturas_df['IRPF'] = 0.0
            
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
                    total_cuota_empresarial as 'Total Cuota Empresarial',
                    comments as 'Comments'
                FROM nominas
            """, conn)
            
            tabla_nominas_df = pd.read_sql_query("""
                SELECT 
                    id_documento as 'ID Documento',
                    descripcion as 'Descripción',
                    devengos as 'Devengos',
                    deducciones as 'Deducciones'
                FROM nominas_tabla
            """, conn)
            
            # Create Excel file in memory with all sheets using xlsxwriter
            excel_file = BytesIO()
            
            try:
                with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                    # Create a default sheet
                    empty_df = pd.DataFrame({'Data': ['No data available']})
                    empty_df.to_excel(writer, sheet_name='Info', index=False)
                    
                    if not facturas_df.empty:
                        facturas_df.to_excel(writer, sheet_name='Facturas', index=False)
                        facturas_tabla_df.to_excel(writer, sheet_name='Facturas Desglose', index=False)
                    if not pagos_df.empty:
                        pagos_df.to_excel(writer, sheet_name='Pagos', index=False)
                        pagos_tabla_df.to_excel(writer, sheet_name='Pagos Desglose', index=False)
                    if not nominas_df.empty or not tabla_nominas_df.empty:
                        nominas_df.to_excel(writer, sheet_name='Nominas', index=False)
                        tabla_nominas_df.to_excel(writer, sheet_name='Nominas Desglose', index=False)

                    # Auto-adjust columns width for all sheets
                    for sheet_name in writer.sheets:
                        worksheet = writer.sheets[sheet_name]
                        for i, col in enumerate(empty_df.columns if sheet_name == 'Info' else 
                                              facturas_df.columns if sheet_name == 'Facturas' else
                                              facturas_tabla_df.columns if sheet_name == 'Facturas Desglose' else
                                              pagos_df.columns if sheet_name == 'Pagos' else
                                              pagos_tabla_df.columns if sheet_name == 'Pagos Desglose' else
                                              nominas_df.columns if sheet_name == 'Nominas' else
                                              tabla_nominas_df.columns):
                            # Get the maximum length of the column
                            max_len = max(
                                len(str(col)),  # Column name length
                                empty_df[col].astype(str).map(len).max() if sheet_name == 'Info' else
                                facturas_df[col].astype(str).map(len).max() if sheet_name == 'Facturas' and col in facturas_df else
                                facturas_tabla_df[col].astype(str).map(len).max() if sheet_name == 'Facturas Desglose' and col in facturas_tabla_df else
                                pagos_df[col].astype(str).map(len).max() if sheet_name == 'Pagos' and col in pagos_df else
                                pagos_tabla_df[col].astype(str).map(len).max() if sheet_name == 'Pagos Desglose' and col in pagos_tabla_df else
                                nominas_df[col].astype(str).map(len).max() if sheet_name == 'Nominas' and col in nominas_df else
                                tabla_nominas_df[col].astype(str).map(len).max() if sheet_name == 'Nominas Desglose' and col in tabla_nominas_df else 0
                            )
                            # Set the column width
                            worksheet.set_column(i, i, min(max_len + 2, 50))
                
                writer.close()
            except Exception as excel_error:
                print(f"Excel generation error: {str(excel_error)}")
                raise HTTPException(status_code=500, detail=f"Failed to generate Excel file: {str(excel_error)}")
            
            excel_file.seek(0)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"autoaudit_{timestamp}.xlsx"
            
            return Response(
                content=excel_file.getvalue(),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
            
    except Exception as e:
        print(f"Excel download error: {str(e)}")
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
        raise HTTPException(status_code=500, detail=str(e))

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
    """Process a PDF file by converting it to PNG images
    
    This endpoint accepts a PDF file (or ZIP of PDFs), extracts/converts them to 
    PNGs and saves them to the appropriate directory for processing.
    The original PDFs are not stored, only the resulting PNG images.
    """
    try:
        # Create temp directory for processing
        temp_dir = Path("temp_pdf_processing")
        output_dir = Path("images")
        temp_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)
        
        filename = file.filename
        file_ext = Path(filename).suffix.lower()
        
        # Save uploaded file to temp directory
        file_path = temp_dir / filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        results = []
        
        if file_ext == ".pdf":
            # Process single PDF
            png_files = process_pdf_to_png(file_path, output_dir)
            
            # Store images in database and collect results
            for png_file in png_files:
                image_data = {
                    "filename": str(png_file.name),
                    "size": png_file.stat().st_size,
                    "path": str(png_file),
                    "processed": False,
                    "source_pdf": filename,
                    "upload_date": datetime.now().isoformat()
                }
                # Store in database
                await store_image_in_db(image_data)
                results.append(image_data)
                
        elif file_ext == ".zip":
            # Process ZIP containing PDFs
            import zipfile
            import shutil
            
            # Extract zip
            extract_dir = temp_dir / "extracted"
            extract_dir.mkdir(exist_ok=True)
            
            # Track valid and invalid files
            valid_files = []
            invalid_files = []
            
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Process all PDFs in the extracted directory
            for file_path in extract_dir.glob("**/*.*"):
                if file_path.suffix.lower() == ".pdf":
                    valid_files.append(file_path)
                else:
                    invalid_files.append(file_path)
            
            # Log invalid files
            if invalid_files:
                logger.warning(f"Ignored {len(invalid_files)} non-PDF files in ZIP archive: {[f.name for f in invalid_files]}")
            
            # Process valid PDF files
            for pdf_file in valid_files:
                png_files = process_pdf_to_png(pdf_file, output_dir)
                
                # Store images in database and collect results
                for png_file in png_files:
                    image_data = {
                        "filename": str(png_file.name),
                        "size": png_file.stat().st_size,
                        "path": str(png_file),
                        "processed": False,
                        "source_pdf": pdf_file.name,
                        "upload_date": datetime.now().isoformat()
                    }
                    # Store in database
                    await store_image_in_db(image_data)
                    results.append(image_data)
            
            # Clean up extracted directory
            shutil.rmtree(extract_dir, ignore_errors=True)
        
        else:
            raise HTTPException(status_code=400, detail="Uploaded file must be a PDF or ZIP containing PDFs")
        
        # Clean up temp file
        file_path.unlink(missing_ok=True)
        
        return results
    
    except Exception as e:
        logger.error(f"Error processing PDF upload: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    finally:
        # Clean up temp directory if empty
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()

@app.post("/upload/pdf/{doc_type}")
async def upload_pdf_by_type(
    doc_type: Literal["facturas", "pagos", "nominas"],
    file: UploadFile = File(...)
):
    """Process a PDF file and store it according to document type
    
    This endpoint accepts a PDF file (or ZIP of PDFs), converts them to PNGs,
    and saves them to the appropriate directory based on document type.
    The original PDFs are not stored, only the resulting PNG images.
    
    Args:
        doc_type: Type of document (facturas, pagos, nominas)
        file: Uploaded PDF or ZIP file
    """
    try:
        # Create temp directory for processing
        temp_dir = Path("temp_pdf_processing")
        
        # Get appropriate output directory based on doc_type
        if doc_type == "facturas":
            output_dir = IMAGES_DIR / "facturas"
        elif doc_type == "pagos":
            output_dir = IMAGES_DIR / "pagos"
        elif doc_type == "nominas":
            output_dir = IMAGES_DIR / "nominas"
        else:
            raise HTTPException(status_code=400, detail="Invalid document type")
        
        # Create directories
        temp_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Log directory information
        logger.info(f"Document type: {doc_type}, Output directory: {output_dir}")
        logger.info(f"IMAGES_DIR is set to: {IMAGES_DIR}")
        
        # Verify directory exists and has write permissions
        try:
            test_file = output_dir / "test_write_permission.tmp"
            with open(test_file, 'w') as f:
                f.write("test")
            test_file.unlink()
            logger.info(f"Output directory {output_dir} exists and has write permissions")
        except Exception as e:
            logger.error(f"Error writing to output directory {output_dir}: {str(e)}")
            # Continue anyway, as the error will be caught later if needed
        
        filename = file.filename
        file_ext = Path(filename).suffix.lower()
        file_size = len(await file.read())
        file_size_kb = f"{file_size / 1024:.2f} KB"
        
        # Reset file position after reading
        await file.seek(0)
        
        # Save uploaded file to temp directory
        file_path = temp_dir / filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        results = []
        
        if file_ext == ".pdf":
            # Process single PDF
            logger.info(f"Processing single PDF {filename} to {output_dir}")
            png_files = process_pdf_to_png(file_path, output_dir)
            
            # Store images in database and collect results
            for png_file in png_files:
                image_data = {
                    "filename": str(png_file.name),
                    "size": png_file.stat().st_size,
                    "path": str(png_file),
                    "processed": False,
                    "doc_type": doc_type,
                    "source_pdf": filename,
                    "upload_date": datetime.now().isoformat()
                }
                # Store in database
                await store_image_in_db(image_data, doc_type)
                results.append(image_data)
            
            # Save conversion history
            history_entry = {
                "filename": filename,
                "pages": len(png_files),
                "status": "Success",
                "file_size": file_size_kb,
                "document_type": doc_type
            }
            await save_pdf_conversion_history(history_entry)
                
        elif file_ext == ".zip":
            # Process ZIP containing PDFs
            import zipfile
            import shutil
            
            logger.info(f"Processing ZIP file {filename}")
            
            # Extract zip
            extract_dir = temp_dir / "extracted"
            extract_dir.mkdir(exist_ok=True)
            logger.info(f"Extracting ZIP to {extract_dir}")
            
            # Track valid and invalid files
            valid_files = []
            invalid_files = []
            
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Process all PDFs in the extracted directory
            for file_path in extract_dir.glob("**/*.*"):
                if file_path.suffix.lower() == ".pdf":
                    valid_files.append(file_path)
                    logger.info(f"Found valid PDF: {file_path}")
                else:
                    invalid_files.append(file_path)
                    logger.info(f"Found invalid file: {file_path}")
            
            # Log invalid files
            if invalid_files:
                logger.warning(f"Ignored {len(invalid_files)} non-PDF files in ZIP archive: {[f.name for f in invalid_files]}")
            
            total_png_files = []
            
            logger.info(f"Processing {len(valid_files)} valid PDFs from ZIP, output to {output_dir}")
            
            # Process valid PDF files
            for pdf_file in valid_files:
                logger.info(f"Processing PDF from ZIP: {pdf_file}")
                png_files = process_pdf_to_png(pdf_file, output_dir)
                total_png_files.extend(png_files)
                
                # Store images in database and collect results
                for png_file in png_files:
                    # Verify the file exists
                    if not png_file.exists():
                        logger.warning(f"PNG file doesn't exist after conversion: {png_file}")
                        continue
                        
                    image_data = {
                        "filename": str(png_file.name),
                        "size": png_file.stat().st_size,
                        "path": str(png_file),
                        "processed": False,
                        "doc_type": doc_type,
                        "source_pdf": pdf_file.name,
                        "upload_date": datetime.now().isoformat()
                    }
                    # Store in database
                    await store_image_in_db(image_data, doc_type)
                    results.append(image_data)
                
                # Save conversion history for each PDF
                history_entry = {
                    "filename": pdf_file.name,
                    "pages": len(png_files),
                    "status": "Success",
                    "file_size": f"{pdf_file.stat().st_size / 1024:.2f} KB",
                    "document_type": doc_type
                }
                await save_pdf_conversion_history(history_entry)
            
            # Perform filesystem check before cleanup
            output_dir_files = list(output_dir.glob("*.png"))
            logger.info(f"Files in output directory {output_dir} after conversion: {len(output_dir_files)}")
            
            # Clean up extracted directory
            logger.info(f"Cleaning up extracted directory: {extract_dir}")
            shutil.rmtree(extract_dir, ignore_errors=True)
            
            # Save zip conversion history
            if valid_files:
                history_entry = {
                    "filename": filename,
                    "pages": len(total_png_files),
                    "status": f"Success: Extracted {len(valid_files)} PDFs to {len(total_png_files)} images",
                    "file_size": file_size_kb,
                    "document_type": doc_type
                }
                await save_pdf_conversion_history(history_entry)
            else:
                history_entry = {
                    "filename": filename,
                    "pages": 0,
                    "status": "Failed: No valid PDFs found in ZIP",
                    "file_size": file_size_kb,
                    "document_type": doc_type
                }
                await save_pdf_conversion_history(history_entry)
        
        else:
            raise HTTPException(status_code=400, detail="Uploaded file must be a PDF or ZIP containing PDFs")
        
        # Clean up temp file
        file_path.unlink(missing_ok=True)
        
        return results
    
    except Exception as e:
        logger.error(f"Error processing PDF upload for {doc_type}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Save error to conversion history
        try:
            history_entry = {
                "filename": filename if 'filename' in locals() else "unknown",
                "pages": 0,
                "status": f"Failed: {str(e)}",
                "file_size": file_size_kb if 'file_size_kb' in locals() else "0 KB",
                "document_type": doc_type
            }
            await save_pdf_conversion_history(history_entry)
        except Exception as hist_err:
            logger.error(f"Error saving error history: {str(hist_err)}")
            
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    finally:
        # Clean up temp directory if empty
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()

# Add endpoint to delete a specific image by filename and document type
@app.delete("/images/{doc_type}/{filename}")
async def delete_image(doc_type: Literal["facturas", "pagos", "nominas"], filename: str):
    """Delete an image file and its database record
    
    Args:
        doc_type: Type of document (facturas, pagos, nominas)
        filename: Name of the file to delete
    """
    try:
        # Construct file path
        file_path = IMAGES_DIR / doc_type / filename
        table_name = f"images_{doc_type}"
        
        # Delete file from filesystem if it exists
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted image file: {file_path}")
        
        # Delete from database
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {table_name} WHERE filename = ?", (filename,))
            
            deleted_count = cursor.rowcount
            conn.commit()
        
        if deleted_count > 0 or file_path.exists():
            return {"success": True, "message": f"Image {filename} deleted successfully"}
        else:
            return {"success": False, "message": f"Image {filename} not found or already deleted"}
            
    except Exception as e:
        logger.error(f"Error deleting image {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def store_image_in_db(image_data: dict, doc_type: str = None):
    """Store image metadata in the database
    
    Args:
        image_data: Dictionary containing image metadata
        doc_type: Optional document type for categorization
    """
    try:
        # Create tables if they don't exist
        await init_image_tables()
        
        # Use the appropriate table based on doc_type
        table_name = f"images_{doc_type}" if doc_type else "images"
        
        # Extract required fields to match database schema
        filename = image_data.get("filename", "")
        size = image_data.get("size", 0)
        
        # Get file content from path if available
        path = image_data.get("path", "")
        source_pdf = image_data.get("source_pdf", "")
        
        # Read file content if path exists
        content = b''
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                content = f.read()
            logger.info(f"Read file content from {path}, size: {len(content)} bytes")
        else:
            logger.warning(f"File {path} does not exist, storing record with empty content")
        
        # Determine mime type
        mime_type = "image/png"
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
            mime_type = "image/jpeg"
        
        # Connect to database
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Insert into database with correct schema
            cursor.execute(f"""
                INSERT INTO {table_name} 
                (filename, content, size, mime_type)
                VALUES (?, ?, ?, ?)
            """, (filename, content, size, mime_type))
            
            conn.commit()
        
        logger.info(f"Stored image {filename} in {table_name} table")
        
        # Ensure file exists in filesystem
        if doc_type and not (path and os.path.exists(path)) and content:
            # Determine filesystem path
            fs_dir = IMAGES_DIR / doc_type
            fs_dir.mkdir(exist_ok=True, parents=True)
            fs_path = fs_dir / filename
            
            # Only write if doesn't exist or is empty
            if not fs_path.exists() or fs_path.stat().st_size == 0:
                try:
                    with open(fs_path, 'wb') as f:
                        f.write(content)
                    logger.info(f"Restored file {filename} to filesystem at {fs_path}")
                except Exception as fs_err:
                    logger.error(f"Error restoring file {filename} to filesystem: {str(fs_err)}")
        
    except Exception as e:
        logger.error(f"Error storing image in database: {str(e)}")
        # Continue without failing - we'll still return the results
        pass

async def init_image_tables():
    """Initialize image tables in database if they don't exist"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Create general images table if it doesn't exist yet
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL UNIQUE,
                    content BLOB NOT NULL,
                    size INTEGER NOT NULL,
                    mime_type TEXT NOT NULL,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create document-type specific tables matching existing schema
            for doc_type in ["facturas", "pagos", "nominas"]:
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS images_{doc_type} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL UNIQUE,
                        content BLOB NOT NULL,
                        size INTEGER NOT NULL,
                        mime_type TEXT NOT NULL,
                        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            
            # Create PDF conversion history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pdf_conversion_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    filename TEXT NOT NULL,
                    pages INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    file_size TEXT NOT NULL,
                    document_type TEXT NOT NULL
                )
            """)
            
            conn.commit()
        
    except Exception as e:
        logger.error(f"Error initializing image tables: {str(e)}")
        raise

def process_pdf_to_png(pdf_path: Path, output_dir: Path) -> List[Path]:
    """Convert a PDF file to PNG images using pdf2image
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save PNG files
        
    Returns:
        List of paths to generated PNG files
    """
    from pdf2image import convert_from_path
    import PyPDF2
    import math
    
    # Constants
    MAX_PAGES_PER_CHUNK = 100
    
    output_files = []
    
    try:
        # Ensure output directory exists
        output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Converting PDF {pdf_path} to PNGs in {output_dir}")
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            base_filename = pdf_path.stem
            
            # For large PDFs, process in chunks
            if total_pages > MAX_PAGES_PER_CHUNK:
                chunk_temp_dir = Path("temp_pdf_chunks")
                chunk_temp_dir.mkdir(exist_ok=True)
                
                num_chunks = math.ceil(total_pages / MAX_PAGES_PER_CHUNK)
                
                for chunk in range(num_chunks):
                    start_page = chunk * MAX_PAGES_PER_CHUNK
                    end_page = min((chunk + 1) * MAX_PAGES_PER_CHUNK, total_pages)
                    
                    pdf_writer = PyPDF2.PdfWriter()
                    
                    # Add pages for this chunk
                    for page_num in range(start_page, end_page):
                        pdf_writer.add_page(pdf_reader.pages[page_num])
                    
                    # Generate output filename
                    chunk_filename = f"{base_filename}_part{chunk + 1}.pdf"
                    chunk_path = chunk_temp_dir / chunk_filename
                    
                    # Save the chunk
                    with open(chunk_path, 'wb') as output_file:
                        pdf_writer.write(output_file)
                    
                    # Convert chunk to images
                    images = convert_from_path(chunk_path)
                    
                    # Save images
                    for i, image in enumerate(images, start_page + 1):
                        output_filename = f"{base_filename}({i}).png"
                        output_path = output_dir / output_filename
                        image.save(output_path, "PNG")
                        output_files.append(output_path)
                    
                    # Clean up chunk
                    chunk_path.unlink(missing_ok=True)
                
                # Clean up chunk directory
                if chunk_temp_dir.exists() and not any(chunk_temp_dir.iterdir()):
                    chunk_temp_dir.rmdir()
            else:
                # For smaller PDFs, convert directly
                images = convert_from_path(pdf_path)
                
                # Save images
                for i, image in enumerate(images, 1):
                    output_filename = f"{base_filename}({i}).png"
                    output_path = output_dir / output_filename
                    image.save(output_path, "PNG")
                    output_files.append(output_path)
        
        return output_files
        
    except Exception as e:
        logger.error(f"Error converting PDF {pdf_path}: {str(e)}")
        raise

@app.get("/excel/test-download")
async def test_download_excel():
    """Test endpoint for Excel download with minimal functionality"""
    try:
        # Create a simple DataFrame
        test_df = pd.DataFrame({
            'Column1': ['Test1', 'Test2', 'Test3'],
            'Column2': [1, 2, 3],
            'Column3': [4.5, 5.5, 6.5]
        })
        
        # Create Excel file in memory
        excel_file = BytesIO()
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            test_df.to_excel(writer, sheet_name='TestSheet', index=False)
        
        excel_file.seek(0)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"test_{timestamp}.xlsx"
        
        return Response(
            content=excel_file.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def save_pdf_conversion_history(history_data: dict):
    """Save PDF conversion history to database
    
    Args:
        history_data: Dictionary containing history metadata
    """
    try:
        # Create tables if they don't exist
        await init_image_tables()
        
        # Extract required fields
        filename = history_data.get("filename", "")
        pages = history_data.get("pages", 0)
        status = history_data.get("status", "Unknown")
        file_size = history_data.get("file_size", "0 KB")
        document_type = history_data.get("document_type", "General")
        
        # Connect to database
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Insert into database
            cursor.execute("""
                INSERT INTO pdf_conversion_history 
                (filename, pages, status, file_size, document_type)
                VALUES (?, ?, ?, ?, ?)
            """, (filename, pages, status, file_size, document_type))
            
            conn.commit()
            
        logger.info(f"Saved PDF conversion history for {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving PDF conversion history: {str(e)}")
        return False

async def get_pdf_conversion_history():
    """Get PDF conversion history from database
    
    Returns:
        List of PDF conversion history records
    """
    try:
        # Create tables if they don't exist
        await init_image_tables()
        
        # Connect to database
        with sqlite3.connect(DB_PATH) as conn:
            # Configure connection to return rows as dictionaries
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get history records
            cursor.execute("""
                SELECT * FROM pdf_conversion_history 
                ORDER BY timestamp DESC
            """)
            
            # Convert to list of dictionaries
            history = [dict(row) for row in cursor.fetchall()]
            
        return history
        
    except Exception as e:
        logger.error(f"Error getting PDF conversion history: {str(e)}")
        return []

async def clear_pdf_conversion_history():
    """Clear all PDF conversion history records from database
    
    Returns:
        Boolean indicating success or failure
    """
    try:
        # Connect to database
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Delete all history records
            cursor.execute("DELETE FROM pdf_conversion_history")
            
            conn.commit()
        
        logger.info("Cleared PDF conversion history")
        return True
        
    except Exception as e:
        logger.error(f"Error clearing PDF conversion history: {str(e)}")
        return False

async def delete_pdf_conversion_record(record_id: int):
    """Delete a single PDF conversion history record from database
    
    Args:
        record_id: ID of the record to delete
        
    Returns:
        Boolean indicating success or failure
    """
    try:
        # Connect to database
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Check if record exists
            cursor.execute("SELECT id FROM pdf_conversion_history WHERE id = ?", (record_id,))
            if not cursor.fetchone():
                logger.warning(f"PDF conversion record with ID {record_id} not found")
                return False
            
            # Delete the record
            cursor.execute("DELETE FROM pdf_conversion_history WHERE id = ?", (record_id,))
            
            conn.commit()
        
        logger.info(f"Deleted PDF conversion history record with ID {record_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error deleting PDF conversion history record: {str(e)}")
        return False

@app.post("/pdf/history/save")
async def save_conversion_history(history_data: dict):
    """Save PDF conversion history to database"""
    try:
        result = await save_pdf_conversion_history(history_data)
        if result:
            return {"success": True, "message": "Conversion history saved"}
        else:
            return {"success": False, "message": "Failed to save conversion history"}
    except Exception as e:
        logger.error(f"Error saving conversion history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pdf/history")
async def get_conversion_history():
    """Get PDF conversion history from database"""
    try:
        history = await get_pdf_conversion_history()
        return history
    except Exception as e:
        logger.error(f"Error getting conversion history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/pdf/history/clear")
async def clear_conversion_history():
    """Clear all PDF conversion history from database"""
    try:
        result = await clear_pdf_conversion_history()
        if result:
            return {"success": True, "message": "Conversion history cleared"}
        else:
            return {"success": False, "message": "Failed to clear conversion history"}
    except Exception as e:
        logger.error(f"Error clearing conversion history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/pdf/history/{record_id}")
async def delete_conversion_record(record_id: int):
    """Delete a single PDF conversion history record by ID"""
    try:
        result = await delete_pdf_conversion_record(record_id)
        if result:
            return {"success": True, "message": f"Record {record_id} deleted successfully"}
        else:
            return {"success": False, "message": f"Record {record_id} not found or could not be deleted"}
    except Exception as e:
        logger.error(f"Error deleting conversion record: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images/{doc_type}/download/{filename}")
async def download_image(doc_type: Literal["facturas", "pagos", "nominas"], filename: str):
    """Download a single converted PNG image"""
    try:
        # Construct file path
        file_path = IMAGES_DIR / doc_type / filename
        
        # Check if file exists in filesystem
        if file_path.exists():
            return Response(
                content=file_path.read_bytes(),
                media_type="image/png",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}"
                }
            )
        
        # If not in filesystem, try to get from database
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT content, mime_type FROM images_{doc_type} WHERE filename = ?", (filename,))
            row = cursor.fetchone()
            
            if row and row[0]:
                content, mime_type = row
                return Response(
                    content=content,
                    media_type=mime_type or "image/png",
                    headers={
                        "Content-Disposition": f"attachment; filename={filename}"
                    }
                )
            
        raise HTTPException(status_code=404, detail=f"Image {filename} not found")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images/{doc_type}/download-all")
async def download_all_images(doc_type: Literal["facturas", "pagos", "nominas"]):
    """Download all converted images as a ZIP archive"""
    try:
        # Create in-memory ZIP file
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Try filesystem first
            images_dir = IMAGES_DIR / doc_type
            if images_dir.exists():
                for file_path in images_dir.glob("*.png"):
                    zip_file.write(file_path, file_path.name)
            
            # Also check database
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT filename, content FROM images_{doc_type}")
                
                for filename, content in cursor.fetchall():
                    if content:  # Only add if content exists
                        zip_file.writestr(filename, content)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_filename = f"converted_images_{doc_type}_{timestamp}.zip"
        
        # Set buffer position to start
        zip_buffer.seek(0)
        
        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={zip_filename}"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("flow.backend.api:app", host="0.0.0.0", 
                port=int(os.getenv("SERVER_PORT", "8000")), 
                reload=True) 