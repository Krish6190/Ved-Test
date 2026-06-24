
import os
from .vector_engine import LocalVectorDB

rag_db = LocalVectorDB()

__all__ = ["rag_db"]
