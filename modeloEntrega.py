import os
import pandas as pd
import pdfkit
import locale
from jinja2 import Environment, FileSystemLoader
import numpy as np

# Configurar la localización para que use el formato europeo
locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')

# Configure wkhtmltopdf path
config = pdfkit.configuration()

# Función para formatear valores
def format_value(value, column, currency_columns):
    if pd.isna(value):
        return ""
    if column in currency_columns:
        try:
            return locale.currency(float(value), grouping=True)
        except (ValueError, TypeError):
            return str(value)
    if isinstance(value, (float, np.float64)) and value.is_integer():
        return int(value)
    return str(value)

# Configurar Jinja2 y registrar el filtro
env = Environment(loader=FileSystemLoader('templates'))
env.filters['format_value'] = format_value  # Register the filter first
template = env.get_template('report_template.html')  # Then get the template

# Cargar el archivo Excel
df = pd.read_excel("C:/Users/Sergio/GT/AutoAudit/Datos_PDFs_InformesA.xlsx")

# Eliminar las columnas repetidas para el encabezado único
repeated_columns = ['Comunidad Autónoma', 'Nº de informe', 'NIF/CIFBeneficiario', 'Nombre Empresa']

# Definir columnas para incidencias (necesarias para el texto pero no para la tabla)
incidences_columns = ['Observaciones', 'Tipo de incidencia1',
                     'Tipo de Incidencia2', 'Tipo de incidencia3']

# Columnas a excluir de la tabla
columns_to_exclude = incidences_columns + repeated_columns

# Filtrar las columnas antes de procesar los datos
columns_to_keep = [col for col in df.columns if col not in columns_to_exclude]

# Obtener la lista de NIF/CIFBeneficiario únicos
nifs = df['NIF/CIFBeneficiario'].unique()

# Definir la ruta de la carpeta donde guardar los PDFs
pdf_folder = "C:/Users/Sergio/GT/AutoAudit/Datos_PDFs_InformesA"
os.makedirs(pdf_folder, exist_ok=True)

# Listar las columnas de moneda
currency_columns = [
    'Importe en € (Sin IVA)', 
    'Documentación no elegible', 
    'Documentación aceptada', 
    'Pendiente de justificar'
]

# Configuración de pdfkit
options = {
    'page-size': 'A4',
    'orientation': 'Landscape',
    'margin-top': '10mm',
    'margin-right': '10mm',
    'margin-bottom': '10mm',
    'margin-left': '10mm',
    'encoding': 'UTF-8',
    'no-outline': None
}

def create_pdf(data, nif, header_info):
    # Preparar datos de incidencias
    incidencias_data = []
    for _, row in data.iterrows():
        if pd.notna(row.get('Observaciones')):
            incidencias_data.append(f"NºGasto {row['NºGasto']}: {row['Observaciones']}")
        # for i in range(1, 4):
        #     tipo_col = f'Tipo de incidencia{i}'
        #     inc_col = f'Incidencia{i}'
        #     if tipo_col in row and inc_col in row and pd.notna(row.get(tipo_col)) and pd.notna(row.get(inc_col)):
        #         incidencias_data.append(f"NºGasto {row['NºGasto']}: {row[tipo_col]} - {row[inc_col]}")

    # Crear una versión filtrada para la tabla
    table_data = data[columns_to_keep]

    # Renderizar el template HTML
    html_content = template.render(
        header_info=header_info,
        columns=table_data.columns,  # Use filtered columns for table header
        data=table_data,  # Use filtered data for table
        currency_columns=currency_columns,
        incidencias_data=incidencias_data  # Use full data for incidencias
    )

    # Guardar como PDF
    output_file = os.path.join(pdf_folder, f"PDF_{nif}.pdf")
    pdfkit.from_string(html_content, output_file, options=options, configuration=config)

# Generar PDFs
for nif in nifs:
    # Obtener todos los datos primero (incluyendo Observaciones)
    filtered_data_full = df[df['NIF/CIFBeneficiario'] == nif]
    
    # Preparar el encabezado único
    header_info = {}
    for col in repeated_columns:
        if col in filtered_data_full.columns:
            value = filtered_data_full[col].iloc[0]
            if col == 'Nº de informe' and pd.notna(value) and isinstance(value, float) and value.is_integer():
                value = int(value)
            header_info[col] = value
    
    # Crear una versión filtrada para la tabla
    filtered_data = filtered_data_full[columns_to_keep]
    create_pdf(filtered_data_full, nif, header_info)  # Usar filtered_data_full para mantener acceso a Observaciones

print("PDFs generados exitosamente.")