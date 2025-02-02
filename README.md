# OAPIA (Outil d'Assistance Pédagogique d'Intelligence Artificielle)

## Installation
Clonez le dépôt localement:

`git clone https://github.com/ComCyberGEND-IA/OAPIA.git`

- Installez les modules nécessaires :

  pip install -r requirements.txt


## Utilisation classique
Exécutez le script principal (dans le terminale) :

`streamlit run main.py`

Dans l'éditeur :

Appuyer sur le bouton run.

## Utilisation recommandée (avec Docker)

Vous trouverez tous les fichiers nécéssaire pour l'installation dans **Installation_Docker**.

Mettez dans le dossier souhaité tous les fichiers présent dans **Installation_Docker**.

Naviguer vers le fichier en question, dans un terminale : `cd ...OAPIA\Installation_Docker`
Puis run : `docker build -t nom_de_l'image_souhaitée .`.

Une fois fait: 

Toujours dans le terminale: `docker run --gpus all --rm -it -p 8501:8501 -v chemin_du_dossier_souhaitée:/workspace nom_de_l'image`. 

Ajouter les fichiers python dans le dossier (si ce n'est pas déjà le cas).


Puis run: `streamlit run main.py`


## Prérequis avant lancement programme

Cuda d'installer, GPU assez puissant (12Go minimum)

Deux modèles requis : Llama 3 et Phi 3 Vision.

Ceux utilisés par défault dans le programme sont trouvables:

- Llama3 8b Instruct: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
- Phi 3 Vision 128k Instruct: https://huggingface.co/microsoft/Phi-3-vision-128k-instruct

**⚠️Les modèles doivent être clonés dans le même dossier que celui où a été créée l'image**
