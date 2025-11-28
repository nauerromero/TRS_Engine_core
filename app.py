"""
SAORI AI Core - Flask App Entry Point
Railway necesita un archivo app.py o main.py en la raíz para detectar Flask automáticamente
"""
import sys
import os
from pathlib import Path

# Agregar src/ y project root al path
project_root = Path(__file__).parent
src_path = project_root / 'src'

# Verificar que src/ existe
if not src_path.exists():
    print(f"[ERROR] src/ directory not found at {src_path}")
    print(f"[DEBUG] Current directory: {os.getcwd()}")
    print(f"[DEBUG] Project root: {project_root}")
    print(f"[DEBUG] Contents of project root: {list(project_root.iterdir())}")
    sys.exit(1)

# Verificar que Modules/ existe
modules_path = project_root / 'Modules'
if not modules_path.exists():
    print(f"[ERROR] Modules/ directory not found at {modules_path}")
    print(f"[DEBUG] Contents of project root: {list(project_root.iterdir())}")
    sys.exit(1)

# Agregar project root y src/ al path de Python
# project_root primero para que Modules/ sea accesible
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))
print(f"[INFO] Added {project_root} to Python path")
print(f"[INFO] Added {src_path} to Python path")

# Verificar que whatsapp_bot.py existe
whatsapp_bot_path = src_path / 'whatsapp_bot.py'
if not whatsapp_bot_path.exists():
    print(f"[ERROR] whatsapp_bot.py not found at {whatsapp_bot_path}")
    print(f"[DEBUG] Contents of src/: {list(src_path.iterdir())}")
    sys.exit(1)

# Importar y ejecutar el bot
try:
    from whatsapp_bot import app
    print("[INFO] Successfully imported whatsapp_bot")
except ImportError as e:
    print(f"[ERROR] Failed to import whatsapp_bot: {e}")
    print(f"[DEBUG] Python path: {sys.path}")
    sys.exit(1)

if __name__ == '__main__':
    # Railway asigna PORT automáticamente
    port = int(os.environ.get('PORT', 5000))
    print(f"[INFO] Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
