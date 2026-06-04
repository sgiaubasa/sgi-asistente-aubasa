-- Tabla principal de datos (reemplaza todos los JSON locales)
CREATE TABLE IF NOT EXISTS data_store (
  key TEXT PRIMARY KEY,
  value JSONB NOT NULL DEFAULT '[]',
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Tabla de usuarios para el login
CREATE TABLE IF NOT EXISTS usuarios_sgi (
  id SERIAL PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  rol TEXT NOT NULL DEFAULT 'usuario',  -- 'admin' o 'usuario'
  nombre TEXT,
  activo BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insertar admin inicial (password: aubasa2024)
INSERT INTO usuarios_sgi (email, password_hash, rol, nombre)
VALUES (
  'admin@aubasa.com',
  'aubasa2024',
  'admin',
  'Administrador SGI'
) ON CONFLICT DO NOTHING;

-- Deshabilitar RLS para uso interno (app corre en servidor)
ALTER TABLE data_store DISABLE ROW LEVEL SECURITY;
ALTER TABLE usuarios_sgi DISABLE ROW LEVEL SECURITY;
