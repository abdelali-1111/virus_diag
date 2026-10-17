"""Point d'entrée du projet.

  python main.py        -> lance l'API (http://127.0.0.1:8000/docs)
  python main.py cli    -> lance l'interface clinique en console
"""
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        from app.interface_clinique import main
        main()
    else:
        import uvicorn
        uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=True)
