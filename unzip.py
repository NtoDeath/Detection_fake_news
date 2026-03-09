import zipfile
import shutil
import os
from pathlib import Path

SUPPRIMER_ZIP_APRES_EXTRACTION = False

dossier_racine = Path('.')

fichiers_zip = list(dossier_racine.rglob('*.zip'))

if not fichiers_zip:
    print("Aucun fichier .zip n'a été trouvé dans le projet.")
else:
    print(f"{len(fichiers_zip)} archive(s) trouvée(s). Début de l'extraction...\n")

    for chemin_zip in fichiers_zip:
        dossier_cible = chemin_zip.parent 

        print(f"Extraction de '{chemin_zip.name}' directement dans '{dossier_cible}/'...")
        
        try:
            with zipfile.ZipFile(chemin_zip, 'r') as archive:
                archive.extractall(path=dossier_cible)
            
            # --- NETTOYAGE MAC ---
            dossier_macosx = dossier_cible / "__MACOSX"
            if dossier_macosx.exists() and dossier_macosx.is_dir():
                shutil.rmtree(dossier_macosx)
                
            print(f"Succès pour {chemin_zip.name} !")
            
            if SUPPRIMER_ZIP_APRES_EXTRACTION:
                os.remove(chemin_zip)
                print(f"Le fichier {chemin_zip.name} a été supprimé pour libérer de l'espace.")
                
        except zipfile.BadZipFile:
            print(f"Erreur : Le fichier {chemin_zip.name} est corrompu ou n'est pas une archive valide.")

print("\nToutes les zip sont décompressées.")