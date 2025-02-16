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
from pydantic import BaseModel

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
    FECHA_FACTURA: str
    PROVEEDOR: str
    BASE_IMPONIBLE: str
    CIF_PROVEEDOR: str
    IRPF: str
    IVA: str
    TOTAL_FACTURA: str

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
            "required": ["CIF_CLIENTE", "CLIENTE", "FICHERO", "NUMERO_FACTURA", "FECHA_FACTURA", 
                        "PROVEEDOR", "BASE_IMPONIBLE", "CIF_PROVEEDOR", "IRPF", "IVA", 
                        "TOTAL_FACTURA"]
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

class NominaRow(BaseModel):
    MES: str
    FECHA_INICIO: str
    FECHA_FIN: str
    CIF: str
    TRABAJADOR: str
    NAF: str
    NIF: str
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
    
    model_config = {
        "json_schema_extra": {
            "type": "object",
            "properties": {
                "MES": {"type": "string"},
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
            },
            "required": ["MES", "FECHA_INICIO", "FECHA_FIN", "CIF", "TRABAJADOR", "NAF", "NIF", "CATEGORIA", "ANTIGUEDAD", "CONTRATO", "TOTAL_DEVENGOS", "TOTAL_DEDUCCIONES", "ABSENTISMOS", "BC_TEORICA", "PRORRATA", "BC_CON_COMPLEMENTOS", "TOTAL_SEG_SOCIAL", "BONIFICACIONES_SS_TRABAJADOR", "TOTAL_RETENCIONES", "TOTAL_RETENCIONES_SS", "LIQUIDO_A_PERCIBIR", "A_ABONAR", "TOTAL_CUOTA_EMPRESARIAL"]
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
    IMPORTE_UNIDAD: str
    UNIDAD: str
    DEVENGOS: str
    DEDUCCIONES: str
    
class TablaNominas(BaseModel):
    rows: List[FilaNominas]
    
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
                            "IMPORTE_UNIDAD": {"type": "string"},
                            "UNIDAD": {"type": "string"},
                            "DEVENGOS": {"type": "string"},
                            "DEDUCCIONES": {"type": "string"}
                        },
                        "required": ["DESCRIPCION","IMPORTE_UNIDAD","UNIDAD","DEVENGOS","DEDUCCIONES"]
                    }
                }
            },
            "required": ["rows"]
        }
    }

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
