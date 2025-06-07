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

from typing_extensions import TypedDict, Sequence, Literal, List, Optional
from typing import Annotated
from langchain_core.messages import BaseMessage
import operator
from pydantic import BaseModel, Field, validator, field_validator, model_validator
import re
from datetime import datetime
import locale
import logging

logger = logging.getLogger(__name__)

def validate_nif(nif: str) -> tuple[bool, str]:
    """
    Validates a Spanish NIF/NIE.
    Returns a tuple of (is_valid, error_message).
    If valid, error_message will be empty.
    
    Rules:
    - NIFs must be 9 characters (if 8, pad with leading 0)
    - NIFs start with number or X/Y/Z and end with letter
    """
    if not nif:
        return False, "NIF cannot be empty"
    
    # Remove any whitespace and make uppercase
    nif = nif.strip().upper()
    
    # Basic length check and padding
    if len(nif) == 8:
        nif = '0' + nif
    elif len(nif) != 9:
        return False, f"NIF must be 8 or 9 characters, got {len(nif)}"
    
    # Patterns for different document types
    nif_pattern = r'^[0-9XYZ][0-9]{7}[A-Z]$'  # For NIFs and NIEs
    
    is_nif = bool(re.match(nif_pattern, nif))
    
    if not (is_nif):
        return False, "Invalid format. NIF must start with number/X/Y/Z and end with letter."
    
    return True, ""

def validate_cif(cif: str) -> tuple[bool, str]:
    """
    Validates a Spanish CIF.
    Returns a tuple of (is_valid, error_message).
    If valid, error_message will be empty.
    
    Rules:
    - CIFs must be 9 characters (if 8, pad with leading 0)
    - CIFs start and end with letters
    """
    if not cif:
        return False, "CIF cannot be empty"
    
    # Remove any whitespace and make uppercase
    cif = cif.strip().upper()
    
    # Basic length check and padding
    if len(cif) == 8:
        cif = '0' + cif
    elif len(cif) != 9:
        return False, f"CIF must be 8 or 9 characters, got {len(cif)}"
    
    # Patterns for different document types
    cif_pattern = r'^[ABCDEFGHJKLMNPQRSUVW][0-9]{7}[A-Z]$'  # For CIFs
    
    is_cif = bool(re.match(cif_pattern, cif))
    
    if not (is_cif):
        return False, "Invalid format. CIF must start and end with letters."
    
    return True, ""



def extract_nif_from_text(text: str) -> tuple[str, str]:
    """
    Attempts to extract a valid NIF/CIF from text.
    Returns a tuple of (extracted_nif, error_message).
    If no valid NIF/CIF is found, error_message will contain the reason.
    """
    if not text:
        return "", "No text provided to extract NIF/CIF from"
    
    # Common patterns for NIFs/CIFs in text
    patterns = [
        r'[0-9XYZ][0-9]{6,7}[A-Z]',  # NIF/NIE pattern (7 or 8 digits)
        r'[ABCDEFGHJKLMNPQRSUVW][0-9]{6,7}[A-Z]'  # CIF pattern (7 or 8 digits)
    ]
    
    # Try each pattern
    for pattern in patterns:
        matches = re.finditer(pattern, text.upper())
        for match in matches:
            potential_nif = match.group()
            # Add leading zero if needed
            if len(potential_nif) == 8:
                potential_nif = '0' + potential_nif
            is_valid, error = validate_nif(potential_nif)
            if is_valid:
                return potential_nif, ""
    
    return "", "No valid NIF/CIF found in text"

# Use Pydantic BaseModel class
class DocType(BaseModel):
    TIPO: Literal['Orden de transferencia', 'Transferencia emitida', 'Adeudo por transferencia', 'Orden de pago', 'Detalle movimiento', 'Certificado bancario', 'Tarjeta de credito', 'Extracto movimiento', 'Arqueo de caja']

class Transferencia(BaseModel):
    DOCUMENTO: str
    TIPO: Literal['Orden de transferencia', 'Transferencia emitida', 'Adeudo por transferencia', 'Orden de pago', 'Detalle movimiento', 'Certificado bancario']
    FECHA_VALOR: str
    ORDENANTE: str
    BENEFICIARIO: str
    CONCEPTO: str
    IMPORTE: str

class TarjetaRow(BaseModel):
    DOCUMENTO: str
    TIPO: Literal['Tarjeta de credito', 'Extracto movimiento', 'Arqueo de caja']
    FECHA_VALOR: str
    ORDENANTE: str
    BENEFICIARIO: str | None = None
    CONCEPTO: str
    IMPORTE: str

    
    model_config = {
        "json_schema_extra": {
            "type": "object",
            "properties": {
                "DOCUMENTO": {"type": "string"},
                "TIPO": {"type": "string", "enum": ["Tarjeta de credito", "Extracto movimiento", "Arqueo de caja", "Orden de transferencia", "Transferencia emitida", "Adeudo por transferencia", "Orden de pago", "Detalle movimiento", "Certificado bancario"]},
                "FECHA_VALOR": {"type": "string"},
                "ORDENANTE": {"type": "string"},
                "BENEFICIARIO": {"type": "string"},
                "CONCEPTO": {"type": "string"},
                "IMPORTE": {"type": "string"},
                
            },
            "required": ["DOCUMENTO", "TIPO", "FECHA_VALOR", "ORDENANTE", "BENEFICIARIO", "CONCEPTO", "IMPORTE"]
        }
    }

class Tarjeta(BaseModel):
    rows: List[TarjetaRow]

    model_config = {
        "json_schema_extra": {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "DOCUMENTO": {"type": "string"},
                            "TIPO": {"type": "string", "enum": ["Tarjeta de credito", "Extracto movimiento", "Arqueo de caja", "Orden de transferencia", "Transferencia emitida", "Adeudo por transferencia", "Orden de pago", "Detalle movimiento", "Certificado bancario"]},
                            "FECHA_VALOR": {"type": "string"},
                            "ORDENANTE": {"type": "string"},
                            "BENEFICIARIO": {"type": "string"},
                            "CONCEPTO": {"type": "string"},
                            "IMPORTE": {"type": "string"},
                            
                        },
                        "required": ["DOCUMENTO", "TIPO", "FECHA_VALOR", "ORDENANTE", "BENEFICIARIO", "CONCEPTO", "IMPORTE"]
                    }
                }
            },
            "required": ["rows"]
        }
    }

class FacturaRow(BaseModel):
    CIF_CLIENTE: str
    CLIENTE: str
    FICHERO: str
    NUMERO_FACTURA: str
    FECHA_FACTURA: str = Field(default="", description="Invoice date in format DD/MM/YYYY")
    PROVEEDOR: str
    BASE_IMPONIBLE: str
    CIF_PROVEEDOR: str
    IRPF: str
    IVA: str
    TOTAL_FACTURA: str

    @validator('FECHA_FACTURA')
    def validate_fecha_factura(cls, v):
        """Validate and standardize invoice date format, return empty string if invalid"""
        return standardize_date(v)

    @validator('CIF_CLIENTE', 'CIF_PROVEEDOR')
    def validate_cif(cls, v):
        """Validate CIF format"""
        if not v:
            raise ValueError("CIF cannot be empty")
        
        v = v.strip().upper()
        if len(v) == 8:
            v = '0' + v
        elif len(v) != 9:
            raise ValueError(f"CIF must be 8 or 9 characters, got {len(v)}")
        
        cif_pattern = r'^[ABCDEFGHJKLMNPQRSUVW][0-9]{7}[A-Z]$'
        if not re.match(cif_pattern, v):
            raise ValueError("Invalid CIF format. Must start and end with letters and contain 7 digits")
        
        return v

    model_config = {
        "json_schema_extra": {
            "type": "object",
            "properties": {
                "CIF_CLIENTE": {"type": "string"},
                "CLIENTE": {"type": "string"},
                "FICHERO": {"type": "string"},
                "NUMERO_FACTURA": {"type": "string"},
                "FECHA_FACTURA": {"type": "string"},
                "PROVEEDOR": {"type": "string"},
                "BASE_IMPONIBLE": {"type": "string"},
                "CIF_PROVEEDOR": {"type": "string"},
                "IRPF": {"type": "string"},
                "IVA": {"type": "string"},
                "TOTAL_FACTURA": {"type": "string"},
            },
            "required": ["CIF_CLIENTE", "CLIENTE", "FICHERO", "NUMERO_FACTURA", "FECHA_FACTURA", "PROVEEDOR", "BASE_IMPONIBLE", "CIF_PROVEEDOR", "IRPF", "IVA", "TOTAL_FACTURA"]
        }
    }

class Factura(BaseModel):
    rows: List[FacturaRow]

    model_config = {
        "json_schema_extra": {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "CIF_CLIENTE": {"type": "string"},
                            "CLIENTE": {"type": "string"},
                            "FICHERO": {"type": "string"},
                            "NUMERO_FACTURA": {"type": "string"},
                            "FECHA_FACTURA": {"type": "string"},
                            "PROVEEDOR": {"type": "string"},
                            "BASE_IMPONIBLE": {"type": "string"},
                            "CIF_PROVEEDOR": {"type": "string"},
                            "IRPF": {"type": "string"},
                            "IVA": {"type": "string"},
                            "TOTAL_FACTURA": {"type": "string"}
                        },
                        "required": ["CIF_CLIENTE", "CLIENTE", "FICHERO", "NUMERO_FACTURA", "FECHA_FACTURA", "PROVEEDOR", "BASE_IMPONIBLE", "CIF_PROVEEDOR", "IRPF", "IVA", "TOTAL_FACTURA"]
                    }
                }
            },
            "required": ["rows"]
        }
    }

class FilaFacturas(BaseModel):
    CONCEPTO: str | None = None
    UNIDADES: str | None = None
    IMPORTE: str | None = None
    
    model_config = {
        "json_schema_extra": {
            "type": "object",
                "properties": {
                    "CONCEPTO": {"type": "string"},
                    "UNIDADAES": {"type": "string"},
                    "IMPORTE": {"type": "string"}
                },
                "required": ["CONCEPTO","UNIDADES","IMPORTE"]
        }
    }

class FilaTarjetas(BaseModel):
    CONCEPTO: str | None = None
    FECHA_VALOR: str | None = None
    IMPORTE: str | None = None
    
    model_config = {
        "json_schema_extra": {
            "type": "object",
                "properties": {
                    "CONCEPTO": {"type": "string"},
                    "FECHA_VALOR": {"type": "string"},
                    "IMPORTE": {"type": "string"}
                },
                "required": ["CONCEPTO","FECHA_VALOR","IMPORTE"]
        }
    }

class TablaFacturas(BaseModel):
    rows: List[FilaFacturas]
    
    model_config = {
        "json_schema_extra": {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "CONCEPTO": {"type": "string"},
                            "UNIDADAES": {"type": "string"},
                            "IMPORTE": {"type": "string"}
                        },
                        "required": ["CONCEPTO","UNIDADES","IMPORTE"]
                    }
                }
            },
            "required": ["rows"]
        }
    }

class TablaTarjetas(BaseModel):
    rows: List[FilaTarjetas]
    
    model_config = {
        "json_schema_extra": {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "CONCEPTO": {"type": "string"},
                            "FECHA_VALOR": {"type": "string"},
                            "IMPORTE": {"type": "string"}
                        },
                        "required": ["CONCEPTO","FECHA_VALOR","IMPORTE"]
                    }
                }
            },
            "required": ["rows"]
        }
    }

def standardize_date(date_str: str) -> str:
    """
    Converts various date formats to DD/MM/YYYY.
    Returns empty string if conversion fails.
    """
    if not date_str:
        return ""
    
    date_str = date_str.strip().upper()
    
    # Try different date formats
    formats_to_try = [
        # Common formats
        "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d",
        "%d/%m/%y", "%d-%m-%y",
        # Month name formats (Spanish)
        "%d-%b-%Y", "%d/%b/%Y",
        "%b-%y", "%b/%y",
        "%B-%y", "%B/%y",
        "%d-%B-%Y", "%d/%B/%Y"
    ]
    
    # Set locale to Spanish for month names
    try:
        locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_TIME, 'Spanish_Spain.1252')
        except:
            pass

    for fmt in formats_to_try:
        try:
            parsed_date = datetime.strptime(date_str, fmt)
            return parsed_date.strftime("%d/%m/%Y")
        except ValueError:
            continue
    
    return ""  # Return empty string if no format matches

class NominaRow(BaseModel):
    MES: Literal['ENERO', 'FEBRERO', 'MARZO', 'ABRIL', 'MAYO', 'JUNIO', 'JULIO', 'AGOSTO', 'SEPTIEMBRE', 'OCTUBRE', 'NOVIEMBRE', 'DICIEMBRE', 'N/D']
    FECHA_INICIO: str = Field(default="", description="Start date in format DD/MM/YYYY")
    FECHA_FIN: str = Field(default="", description="End date in format DD/MM/YYYY")
    CIF: str = Field(default="", description="Company CIF number")
    TRABAJADOR: str
    NAF: str
    NIF: str = Field(default="", description="Worker NIF/NIE number")
    CATEGORIA: str
    ANTIGUEDAD: str
    CONTRATO: str
    TOTAL_DEVENGOS: str
    TOTAL_DEDUCCIONES: str
    ABSENTISMOS: str
    BC_TEORICA: str
    PRORRATA: str
    BC_CON_COMPLEMENTOS: str
    TOTAL_SEG_SOCIAL: str
    BONIFICACIONES_SS_TRABAJADOR: str
    TOTAL_RETENCIONES: str
    TOTAL_RETENCIONES_SS: str
    LIQUIDO_A_PERCIBIR: str
    A_ABONAR: str
    TOTAL_CUOTA_EMPRESARIAL: str
    
    @validator('CIF')
    def validate_cif_format(cls, v):
        """Validate CIF format without correcting it - correction happens in agent_node.py"""
        if not v:
            return ""  # Allow empty values
        
        v = v.strip().upper()
        if len(v) == 8:
            v = '0' + v
        elif len(v) != 9:
            # Try to recover by padding or truncating
            if len(v) < 8:
                # Too short to be valid
                logger.warning(f"CIF too short: {v}")
                return v
            elif len(v) > 9:
                # Try to use the first 9 characters
                v = v[:9]
        
        # No correction here - just log and return the original value
        # Correction is centralized in agent_node.py correct_cif_ocr_errors
        cif_pattern = r'^[ABCDEFGHJKLMNPQRSUVW][0-9]{7}[A-Z]$'
        if not re.match(cif_pattern, v):
            logger.warning(f"Non-conforming CIF format: {v}")
        
        return v

    @validator('NIF')
    def validate_nif_format(cls, v):
        """Validate NIF format"""
        if not v:
            return ""  # Allow empty values
        
        v = v.strip().upper()
        if len(v) == 8:
            v = '0' + v
        elif len(v) != 9:
            # Try to recover by padding or truncating
            if len(v) < 8:
                # Too short to be valid
                logger.warning(f"NIF too short: {v}")
                return v
            elif len(v) > 9:
                # Try to use the first 9 characters
                v = v[:9]
        
        # Check against pattern and make semi-relaxed validation
        nif_pattern = r'^[0-9XYZ][0-9]{7}[A-Z]$'
        if not re.match(nif_pattern, v):
            logger.warning(f"Non-conforming NIF format: {v}")
            # Return it anyway - we'll try to fix it downstream
            return v
        
        return v

    @validator('FECHA_INICIO', 'FECHA_FIN')
    def validate_fecha(cls, v):
        """Validate and standardize date format, return empty string if invalid"""
        return standardize_date(v)

    model_config = {
        "json_schema_extra": {
            "type": "object",
            "properties": {
                "MES": {
                    "type": "string",
                    "enum": ['ENERO', 'FEBRERO', 'MARZO', 'ABRIL', 'MAYO', 'JUNIO', 'JULIO', 
                            'AGOSTO', 'SEPTIEMBRE', 'OCTUBRE', 'NOVIEMBRE', 'DICIEMBRE', 'N/D']
                },
                "FECHA_INICIO": {"type": "string"},
                "FECHA_FIN": {"type": "string"},
                "CIF": {"type": "string"},
                "TRABAJADOR": {"type": "string"},
                "NAF": {"type": "string"},
                "NIF": {"type": "string"},
                "CATEGORIA": {"type": "string"},
                "ANTIGUEDAD": {"type": "string"},
                "CONTRATO": {"type": "string"},
                "TOTAL_DEVENGOS": {"type": "string"},
                "TOTAL_DEDUCCIONES": {"type": "string"},
                "ABSENTISMOS": {"type": "string"},
                "BC_TEORICA": {"type": "string"},
                "PRORRATA": {"type": "string"},
                "BC_CON_COMPLEMENTOS": {"type": "string"},
                "TOTAL_SEG_SOCIAL": {"type": "string"},
                "BONIFICACIONES_SS_TRABAJADOR": {"type": "string"},
                "TOTAL_RETENCIONES": {"type": "string"},
                "TOTAL_RETENCIONES_SS": {"type": "string"},
                "LIQUIDO_A_PERCIBIR": {"type": "string"},
                "A_ABONAR": {"type": "string"},
                "TOTAL_CUOTA_EMPRESARIAL": {"type": "string"},
                "COMMENTS": {"type": "string"}
            },
            "required": ["MES", "FECHA_INICIO", "FECHA_FIN", "CIF", "TRABAJADOR", "NAF", "NIF", "CATEGORIA", "ANTIGUEDAD", "CONTRATO", "TOTAL_DEVENGOS", "TOTAL_DEDUCCIONES", "ABSENTISMOS", "BC_TEORICA", "PRORRATA", "BC_CON_COMPLEMENTOS", "TOTAL_SEG_SOCIAL", "BONIFICACIONES_SS_TRABAJADOR", "TOTAL_RETENCIONES", "TOTAL_RETENCIONES_SS", "LIQUIDO_A_PERCIBIR", "A_ABONAR", "TOTAL_CUOTA_EMPRESARIAL", "COMMENTS"]
        }
    }

class Nomina(BaseModel):
    rows: List[NominaRow]
    
    model_config = {
        "json_schema_extra": {
            "type": "object",
            "properties": {
                "rows": {"type": "array", "items": {"type": "object", "properties": {}}}
            }
        }
    }
    
class FilaNominas(BaseModel):
    DESCRIPCION: str
    DEVENGOS: str = Field(default="0")
    DEDUCCIONES: str = Field(default="0")
    
    @field_validator('DEVENGOS', 'DEDUCCIONES')
    def validate_amount(cls, v):
        """Validate and standardize amount to numeric string"""
        if not v or v.strip() == '':
            return "0"
        # Remove currency symbols, spaces, and commas
        v = v.replace('€', '').replace(',', '.').strip()
        try:
            # Convert to float and back to string to standardize format
            amount = float(v)
            return f"{amount:.2f}"
        except ValueError:
            return "0"
    
class TablaNominas(BaseModel):
    rows: List[FilaNominas]
    
    @field_validator('rows')
    def validate_rows(cls, rows):
        """Remove rows where both DEVENGOS and DEDUCCIONES are 0"""
        return [
            row for row in rows 
            if float(row.DEVENGOS) > 0 or float(row.DEDUCCIONES) > 0
        ]
    
    model_config = {
        "json_schema_extra": {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "DESCRIPCION": {"type": "string"},
                            "DEVENGOS": {"type": "string"},
                            "DEDUCCIONES": {"type": "string"}
                        },
                        "required": ["DESCRIPCION","DEVENGOS","DEDUCCIONES"]
                    }
                }
            },
            "required": ["rows"],
            "example": {
                "rows": [
                    {
                        "DESCRIPCION": "Salario Base",
                        "DEVENGOS": "1000.00",
                        "DEDUCCIONES": "0"
                    },
                    {
                        "DESCRIPCION": "IRPF",
                        "DEVENGOS": "0",
                        "DEDUCCIONES": "150.00"
                    }
                ]
            }
        }
    }

class NominaTableRow(BaseModel):
    DESCRIPCION: str
    DEVENGOS: str = Field(default="0")
    DEDUCCIONES: str = Field(default="0")

    @model_validator(mode='after')
    def validate_devengos_deducciones(self) -> 'NominaTableRow':
        """Ensure that if one field has value, the other is zero"""
        devengos = float(self.DEVENGOS or '0')
        deducciones = float(self.DEDUCCIONES or '0')
        
        if devengos > 0 and deducciones > 0:
            raise ValueError(
                f"Row '{self.DESCRIPCION}' cannot have both DEVENGOS ({devengos}) "
                f"and DEDUCCIONES ({deducciones}). Each row must have only one type of value."
            )
        
        return self

class AgentState(TypedDict):
    """State for the agents."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str
    filename: str
    workflowtype: Literal['facturas', 'pagos', 'nominas']
    tarjeta: Optional[str] = None
    factura: Optional[str] = None
    transferencia: Optional[str] = None
    nomina: Optional[str] = None
    tablafacturas: Optional[str] = None
    tablatarjetas: Optional[str] = None
    tablanominas: Optional[str] = None