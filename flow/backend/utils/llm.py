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

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from ..models import DocType, Transferencia, FacturaRow, TablaFacturas, TarjetaRow, TablaTarjetas, NominaRow, TablaNominas

llm1 = ChatOpenAI(model="your-model-name")
llm2 = ChatGroq(model="llama3.2-11b-vision-preview")
llm3 = ChatOllama(model="llama3.2-vision").with_structured_output(DocType, method='json_schema')
llm4 = ChatOpenAI(model="your-model-name")
llm5 = ChatOpenAI(model="your-model-name").with_structured_output(DocType)
llm6 = ChatOpenAI(model="your-model-name")
llm7 = ChatOpenAI(model="your-model-name").with_structured_output(Transferencia)
llm8 = ChatOpenAI(model="llama3.1",base_url="http://localhost:11434/v1")
llm9 = ChatOpenAI(model="your-model-name").with_structured_output(TarjetaRow)
llm10 = ChatOpenAI(model="your-model-name").with_structured_output(FacturaRow)
llm11 = ChatOpenAI(model="your-model-name").with_structured_output(TablaFacturas)
llm12 = ChatOpenAI(model="your-model-name").with_structured_output(TablaTarjetas)
llm13 = ChatOpenAI(model="your-model-name").with_structured_output(NominaRow)
llm14 = ChatOpenAI(model="your-model-name").with_structured_output(TablaNominas)