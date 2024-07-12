# Import nécéssaire
import csv
import fitz 
import os
import re
import streamlit as st
import tempfile
import time
import torch
import warnings

# Permet de supprimer des messages inutiles au moment du chargement des modèles
warnings.filterwarnings("ignore")

from gen_phi3_vision import generation_phi3_vision, parametre_phi3_vision
from gen_llama3 import generation_llama3, parametre_Llama3

# Vider le cache pour être sûr d'utiliser toute la place nécéssaire
torch.cuda.empty_cache()

# Pour calculer l'inference
t0 = time.time()

# Fonction pour formater
def correct_encoding(text):
            corrections = {
                'Ã©': 'é',
                'Ã¨': 'è',
                'Ã¢': 'â',
                'Ã´': 'ô',
                'Ãª': 'ê',
                'Å“': 'œ',
                'Ã': 'à',
                'Ã§': 'ç',
                'Ã¹': 'ù',
                'Ã»': 'û',
                'Ã‰': 'É',
                'Ã€': 'À',
                'â€™': "'",
                'â€“': '-',
                'â€œ': '“',
                'â€': '”',
                'Ã¶': 'ö',
                'Ã¯': 'ï',
                'Ã¼': 'ü',
                'â€¦': '…',
                'Â': '',
            }
            for wrong, correct in corrections.items():
                text = text.replace(wrong, correct)
            return text


# Titre du QCM
st.header("QCM : L'IA au service de la sécurité intérieure", divider=True)
# Mention IA
st.markdown("QCM généré par intelligence artificelle, validé par humain")


# Déclaration variables streamlit 
st.session_state.setdefault('commencer', False)
st.session_state.setdefault('next', False)
st.session_state.setdefault('end', False)
st.session_state.setdefault('quest_num', 0)
st.session_state.setdefault('point', 0)
st.session_state.setdefault('nom', None)
st.session_state.setdefault('prenom', None)
st.session_state.setdefault('end', False)
st.session_state.setdefault('dropped', False)
st.session_state.setdefault('uploaded_files', None)
st.session_state.setdefault('gene', None)
st.session_state.setdefault('main', False)
st.session_state.setdefault('liste', [])
st.session_state.setdefault('dico', {})


# Variables pour phi3 vision
st.session_state.setdefault('para_phi', False)
st.session_state.setdefault('processor', None)
st.session_state.setdefault('model_phi', None)
st.session_state.setdefault('prompt_phi', None)

# Variable pour Llama 3 
st.session_state.setdefault('para_llama3', False)
st.session_state.setdefault('llama_model', None)
st.session_state.setdefault('tokenizer', None)


# Permettre de supprimer le drag and drop : widget permettant de mêttre les fichiers voulues pour générer des questions dessus
if not st.session_state.dropped:
    st.session_state.uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True, type=['pdf','txt'])

# Liste qui récupère tous les fichiers envoyés
files = []


# Entrer dans la boucle streamlit pour la génération pour tous les fichiers mis
if st.session_state.uploaded_files:
    st.session_state.dropped = True

    # Attention: Pour la manipulation des fichiers récupérés, 
    # obligation de rester dans l'instruction "with tempfile.TemporaryDirectory()..." 
    # Créer un répertoire temporaire pour pouvoir manipuler les fichiers envoyé comme des fichiers locaux 
    with tempfile.TemporaryDirectory() as temp_dir:

        # Pour tout les documents envoyés via le drag and drop
        for uploaded_file in st.session_state.uploaded_files:

            # Obtenir le contenu du fichier en mémoire
            bytes_data = uploaded_file.getvalue()
            
            # Définir le chemin temporaire du fichier et ainsi pouvoir le manipuler comme un fichier local 
            file_path = os.path.join(temp_dir, uploaded_file.name)

            
            # Écris le fichier sur le disque
            with open(file_path, 'wb') as f:
                f.write(bytes_data)
            
            # Ajoute le chemin temporaire dans une liste
            files.append(file_path)

            # Une fois que tous les documents envoyés sont traités, la génération commence
            if uploaded_file == st.session_state.uploaded_files[-1]:
                st.session_state.gene = True

        # Si Phi3 vision n'a pas encore été paramêtré alors le paramètrage à lieu 
        if not st.session_state.para_phi:
            # st.spinner permet juste d'ajouteur une animation pendant le chargement de phi3 vision
            with st.spinner('Chargement de phi 3 vision...'):
                st.session_state.processor, st.session_state.model_phi,  st.session_state.prompt_phi, st.session_state.para_phi = parametre_phi3_vision()
                st.success('Modèle chargé avec succès!')

        # Une fois le paramêtrage fait, on génère
        if st.session_state.gene:

            # La génération se fait pour tous les documents
            for file in files:

                # Ouvre le document à l'aide des chemains créer plus tôt
                doc = fitz.open(file)

                # Compte le nombre de page du document actuel
                page_count = doc.page_count

                # Liste des refs déjà traités
                xreflist = []  

                # Liste de toutes les refs des images (voire plus bas)
                imglist = []  

                # Variable pour envoyer les textes à Llama3
                all_pages = []

                # Vide la variable pour permettre d'actualiser le st.write et non pas écrire en dessous
                with st.empty():
                    # Boucle pour traiter chaque page du fichier actuel
                    for pno in range(page_count):

                        # Sert seulement à creer deux colonnes pour la présentation du chargement, rien d'autre
                        gauche, droite = st.columns(2)

                        with droite:
                            st.write(f"Extraction page : {pno+1} / {page_count}")

                        with gauche: 
                            with st.spinner('Extraction des images et du texte en cours...'):
                                # Obtenir la liste des images sur la page
                                images = doc.get_page_images(pno)  

                                # Ajouter les références des images à la liste
                                imglist.extend([x[0] for x in images])  

                                # Variable permettant de récuperer le contenu de la page en texte
                                text =''

                                # Ajoute les descriptions d'images
                                text += generation_phi3_vision(images, doc, xreflist, text, st.session_state.processor, st.session_state.model_phi,  st.session_state.prompt_phi)

                                # Load la page actuel pour permettre d'extraire tout le texte
                                page = doc.load_page(pno)
                                text += page.get_text()

                        # Ajoute chaque page textuelle (description image + texte simple) dans une liste
                        all_pages.append(text)               
                    
                # Joint toutes les pages en un et un seul document pour faciliter la compréhension de Llama3
                document = '\n'.join(all_pages)

                # Retirer les doublons des images trouvées
                imglist = list(set(imglist))  

                # Afficher le nombre total d'images trouvées
                st.write(len(set(imglist)), "images in total")

                # Afficher le nombre total d'images extraites
                st.write(len(xreflist), "images extracted")
                st.write("#### Tous les documents sont extraits ####")
        
                # Voire le temps que la description d'image prend sur tout le document
                mid_time = time.time()
                st.write("Inference time: {}".format(mid_time - t0))

                # Une fois fait, vide le cache pour permettre à Llama3 de prendre toute la place dont il a besoin
                torch.cuda.empty_cache()

                # Paramêtre Llama3 si pas encore fait
                if not st.session_state.para_llama3:

                    # st.spinner permet juste d'ajouteur une animation pendant le chargement de phi3 vision
                    with st.spinner('Chargement de Llama 3...'):
                        st.session_state.llama_model, st.session_state.tokenizer, st.session_state.para_llama3 = parametre_Llama3()
                        st.success('Modèle chargé avec succès!')
                
                # st.spinner permet juste d'ajouteur une animation pendant le génération llama3
                with st.spinner('Génération du QCM...'):
                    # Récupère une liste des générations faites par Llama3
                    st.session_state.liste = generation_llama3(document, st.session_state.llama_model, st.session_state.tokenizer)
                
                # Calcul le temps mis une fois les deux types de générations faites
                end_time = time.time()
                st.write("Inference time: {}".format(end_time - t0))

                # Rassemble toutes les question générées par Llama3 en un seul document
                st.session_state.liste = '\n'.join(st.session_state.liste)
                print("Listes des questions:  ", st.session_state.liste)

                # Permet de rentrer dans la boucle principale après généaration des questions
                st.session_state.main = True

                # Ferme la possibilités de remettre des fichiers
                st.session_state.uploaded_files = False

                # Ferme la génération
                st.session_state.gene = False


# Main loop
if st.session_state.main:

    # Expression permettant de formater
    question = "##Question:..*?\?\n"
    option = r'Options:\n\s*A\.\s*.+\n\s*B\.\s*.+\n\s*C\.\s*.+\n\s*D\.\s*.+'
    reponse =  r'Reponse..\s*\w+'


    # 3 listes créer, une pour les questions, une pour les options et une pour les réponses
    # re.findall renvoie une liste de tout les éléments respectant la regex
    liste_question = re.findall(question, st.session_state.liste)

    # Une fois toutes les questions récupérée, supprime le "##" (servant simplement à formater le modèle)
    for i in range(len(liste_question)):
        liste_question[i] = liste_question[i][len("##"):].strip()

    liste_option = re.findall(option, st.session_state.liste)
    liste_reponse = re.findall(reponse, st.session_state.liste)

    # Initialise la variable pour le rendu final
    liste_finale = []


    # Check chaque question jusqu'à ce qu'une question n'est pas d'option ou de réponse associé
    for i in range(len(liste_question)):

        # Afin de créer une liste contenant question, options, et réponse 
        liste_inte = []

        # Ajoute la question à la liste
        liste_inte.append(liste_question[i])

        # S'il n'y a pas d'options associé à la dernière question, on supprime la question
        try: 
            liste_inte.append(liste_option[i])
        except: 
            liste_inte.pop()
            break

        # S'il n'y a pas de réponse associée à la dernière question mais qu'il y a des options, on supprime la question et les options
        try:
            liste_inte.append(liste_reponse[i]+'\n')
        except: 
            liste_inte.pop()
            liste_inte.pop()
            break   # Si plus d'options ou de réponses, alors la question est incomplète et donc inutile

        # Transforme la liste comprenant les questions, options et réponse. Exemple de sortie de la liste: [question1 options1 reponse1, question2 options2 reponse2, questions3 options3 reponse3, ...]
        liste_finale.append('\n'.join(liste_inte))

    # Réencode les caractères spéciaux en les remplaçant via la fonction correct_encoding
    liste_finale = [correct_encoding(phrase) for phrase in liste_finale]
    
    # Initialisation du dico contenant toutes les questions, options et réponses
    st.session_state.dico = {}

    # Création du dictionnaire
    for indexe, question in enumerate(liste_finale):
        st.session_state.dico[f'question{indexe}']= re.search("Question:..*?\?\n", question)[0]
        st.session_state.dico[f'option{indexe}'] = re.search(option, question)[0].split('\n')
        st.session_state.dico[f'option{indexe}'] = [value.strip() for value in st.session_state.dico[f'option{indexe}'][1:]]
        st.session_state.dico[f'reponse{indexe}'] = re.search(reponse, question)[0]
    
    # Une fois tout cela fait, le QCM peut commencer
    if not st.session_state.commencer:
        st.session_state.nom=st.text_input("Votre nom :")
        st.session_state.prenom=st.text_input("Votre prénom :")
        button_start = st.button(label = "Commencer", key='commence')

        if button_start:
            st.session_state.commencer = True

    if st.session_state.commencer:

        # Permet de passer à la question suivante
        if st.session_state.next:
                    st.session_state.quest_num += 1
                    st.session_state.next = False

        # Selon le nombre de question voulu
        if st.session_state.quest_num <= 10:
            
            st.session_state.repondu = False

            envoyer = st.button('Envoyer')

            #NOTE Faire une condition pour le cas où rien n'est sélectionnée.
            if envoyer and not st.session_state.repondu:
                st.session_state.repondu = True

            quest_num = st.session_state.quest_num+1

            print(st.session_state.dico)

            # Affichage de la question
            question = st.write('**Question ' +str(quest_num) + ' :**\n\n ' , st.session_state.dico[f'question{st.session_state.quest_num}'])
            st.write('Choix: ')

            #BUG Les possibilités ne se change qu'après avoir cliquer sur l'une des possibilités a nouveau
            # Affichage des propositions
            option1 = st.checkbox(label=st.session_state.dico[f'option{st.session_state.quest_num}'][0], key=f'A{st.session_state.quest_num}', disabled=st.session_state.repondu)
            option2 = st.checkbox(label=st.session_state.dico[f'option{st.session_state.quest_num}'][1], key=f'B{st.session_state.quest_num}', disabled=st.session_state.repondu)
            option3 = st.checkbox(label=st.session_state.dico[f'option{st.session_state.quest_num}'][2], key=f'C{st.session_state.quest_num}', disabled=st.session_state.repondu)
            option4 = st.checkbox(label=st.session_state.dico[f'option{st.session_state.quest_num}'][3], key=f'D{st.session_state.quest_num}', disabled=st.session_state.repondu)
            
            # Bouton permettant de passer à la question suivante
            if st.button('Question suivante') :
                st.session_state.next = True


            # Donner un point si bonne réponse
            if (option1 and envoyer) and (st.session_state.dico[f'option{st.session_state.quest_num}'][0][:1].strip() in  st.session_state.dico[f'reponse{st.session_state.quest_num}']):
                st.session_state.point += 1

            elif (option2 and envoyer) and (st.session_state.dico[f'option{st.session_state.quest_num}'][1][:1].strip() in  st.session_state.dico[f'reponse{st.session_state.quest_num}']):
                st.session_state.point += 1

            elif (option3 and envoyer) and (st.session_state.dico[f'option{st.session_state.quest_num}'][2][:1].strip() in  st.session_state.dico[f'reponse{st.session_state.quest_num}']):
                st.session_state.point += 1

            elif (option4 and envoyer) and (st.session_state.dico[f'option{st.session_state.quest_num}'][3][:1].strip() in  st.session_state.dico[f'reponse{st.session_state.quest_num}']):
                st.session_state.point += 1

            # Si la réponse a déjà été envoyée, afficher la réponse correcte
            if st.session_state.repondu:
                rep = st.session_state.dico[f'reponse{st.session_state.quest_num}']
                st.write(f'**Réponse** : {rep}')

            st.markdown(f"<h1 style='font-size: 30px;'> Note : {st.session_state.point}/{len(st.session_state.dico)}</h1>", unsafe_allow_html=True)

        # Une fois toute les question faites, affiche le résultat
        else:
            st.write(f'Le test est fini. Note finale : {st.session_state.point}/{len(st.session_state.dico)}')
            st.session_state.end = True

    # Le QCM étant fini, enregistre le résultat dans un fichier csv
    if st.session_state.end:

        # Charger ou créer le fichier CSV pour enregistrer les résultats
        csv_file = r'/workspace/resultats.csv'

        with open(csv_file, 'w', newline='') as csvfile:
            line = csv.writer(csvfile)
            line.writerow(['Nom', 'Prénom', 'Résultat'])
            line.writerow([st.session_state.nom, st.session_state.prenom,st.session_state.point])
