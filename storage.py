"""
Reemplaza load_json/save_json locales → Supabase
y ChromaDB → Pinecone con embeddings integrados (llama-text-embed-v2)
"""
import os, json
from supabase import create_client

_sb = None

def _supabase():
    global _sb
    if _sb is None:
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")
        if url and key:
            _sb = create_client(url, key)
    return _sb

# ── DATA STORE (reemplaza archivos JSON) ──────────────────────────────────────

def load_data(key: str, default):
    """Lee un valor de Supabase data_store. Fallback a default si no existe."""
    try:
        sb = _supabase()
        if sb is None:
            return default
        res = sb.table("data_store").select("value").eq("key", key).execute()
        if res.data:
            return res.data[0]["value"]
    except Exception as e:
        print(f"[storage] load_data({key}): {e}")
    return default

def save_data(key: str, data) -> bool:
    """Guarda un valor en Supabase data_store (upsert)."""
    try:
        sb = _supabase()
        if sb is None:
            return False
        sb.table("data_store").upsert({
            "key": key,
            "value": data,
            "updated_at": "now()"
        }).execute()
        return True
    except Exception as e:
        print(f"[storage] save_data({key}): {e}")
        return False

# ── VECTOR STORE (reemplaza ChromaDB) ─────────────────────────────────────────

_pc_index = None

def _pinecone_index():
    global _pc_index
    if _pc_index is None:
        try:
            from pinecone import Pinecone
            api_key = os.environ.get("PINECONE_API_KEY", "")
            index_name = os.environ.get("PINECONE_INDEX", "sgi-documentos")
            if api_key:
                pc = Pinecone(api_key=api_key)
                _pc_index = pc.Index(index_name)
        except Exception as e:
            print(f"[storage] Pinecone init error: {e}")
    return _pc_index

def vector_upsert(doc_id: str, chunks: list[str], source: str):
    """Indexa chunks de texto en Pinecone."""
    try:
        idx = _pinecone_index()
        if idx is None:
            return
        records = [
            {"_id": f"{doc_id}_{i}", "text": chunk, "source": source}
            for i, chunk in enumerate(chunks)
        ]
        # Pinecone con integrated embedding acepta lotes de 96
        for i in range(0, len(records), 90):
            idx.upsert_records("__default__", records[i:i+90])
    except Exception as e:
        print(f"[storage] vector_upsert error: {e}")

def vector_query(question: str, top_k: int = 3) -> str:
    """Busca en Pinecone y devuelve texto concatenado de los chunks relevantes."""
    try:
        idx = _pinecone_index()
        if idx is None:
            return ""
        results = idx.search(
            namespace="__default__",
            query={"inputs": {"text": question}, "top_k": top_k},
            fields=["text", "source"]
        )
        hits = results.get("result", {}).get("hits", [])
        if not hits:
            return ""
        partes = []
        for h in hits:
            fields = h.get("fields", {})
            src = fields.get("source", "")
            txt = fields.get("text", "")
            partes.append(f"[{src}]\n{txt}")
        return "\n\n---\n\n".join(partes)
    except Exception as e:
        print(f"[storage] vector_query error: {e}")
        return ""

def vector_delete(doc_id: str):
    """Elimina todos los chunks de un documento del índice."""
    try:
        idx = _pinecone_index()
        if idx is None:
            return
        # Listar IDs que empiezan con doc_id
        idx.delete(filter={"source": {"$eq": doc_id}}, namespace="__default__")
    except Exception as e:
        print(f"[storage] vector_delete error: {e}")

def vector_count() -> int:
    """Retorna cantidad de vectores indexados."""
    try:
        idx = _pinecone_index()
        if idx is None:
            return 0
        stats = idx.describe_index_stats()
        return stats.get("total_vector_count", 0)
    except:
        return 0

# ── AUTH ───────────────────────────────────────────────────────────────────────

def verificar_login(email: str, password: str) -> dict | None:
    """Verifica credenciales. Retorna dict con rol o None si falla."""
    try:
        sb = _supabase()
        if sb is None:
            return None
        res = sb.table("usuarios_sgi") \
            .select("email,rol,nombre,activo") \
            .eq("email", email) \
            .eq("password_hash", password) \
            .eq("activo", True) \
            .execute()
        if res.data:
            return res.data[0]
    except Exception as e:
        print(f"[storage] login error: {e}")
    return None

def listar_usuarios() -> list:
    try:
        sb = _supabase()
        if sb is None:
            return []
        res = sb.table("usuarios_sgi").select("id,email,rol,nombre,activo,created_at").execute()
        return res.data or []
    except:
        return []

def crear_usuario(email: str, password: str, rol: str, nombre: str) -> bool:
    try:
        sb = _supabase()
        if sb is None:
            return False
        sb.table("usuarios_sgi").insert({
            "email": email,
            "password_hash": password,
            "rol": rol,
            "nombre": nombre
        }).execute()
        return True
    except Exception as e:
        print(f"[storage] crear_usuario error: {e}")
        return False

def eliminar_usuario(uid: int) -> bool:
    try:
        sb = _supabase()
        if sb is None:
            return False
        sb.table("usuarios_sgi").delete().eq("id", uid).execute()
        return True
    except:
        return False
