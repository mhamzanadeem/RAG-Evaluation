"""
pipelines/__init__.py
Exports all four RAG pipeline run functions.
Each pipeline takes (query, index, config) and returns a PipelineResult dict.
"""

from src.pipelines.rag_fusion import run as rag_fusion
from src.pipelines.hyde import run as hyde
from src.pipelines.crag import run as crag
from src.pipelines.graph_rag import run as graph_rag

PIPELINE_NAMES = ["rag_fusion", "hyde", "crag", "graph_rag"]

PIPELINES = {
    "rag_fusion": rag_fusion,
    "hyde": hyde,
    "crag": crag,
    "graph_rag": graph_rag,
}
