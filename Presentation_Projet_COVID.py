# -*- coding: utf-8 -*-
"""
PRESENTATION DU PROJET "Covid-19 pulmonary X-ray Analysis"

Pour visualiser le résultat du streamlit :
   cd C:\MesDocuments\DATASCIENTEST\PROJET\Livrable 3 Streamlit\
   streamlit run "Presentation_Projet_COVID.py"
"""


import streamlit as st
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import cv2
from PIL import Image
import seaborn as sns
sns.set_theme()
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from datetime import datetime


#CHARGEMENT DES DONNEES
if 'chargement_model' not in st.session_state:
    st.session_state.chargement_model = False
if 'test_pred_orig' not in st.session_state:
    st.session_state.test_pred_orig = None
if 'test_pred_corr' not in st.session_state:
    st.session_state.test_pred_corr = None
if 'bouton_train' not in st.session_state:
    st.session_state.bouton_train = "Avant"
if 'training_history' not in st.session_state:
    st.session_state.training_history = None


global chemin_csv, chemin_images
global donnees_init_FULL, donnees_800_orig, donnees_800_corr
global data_simu, target_simu
chemin_csv = "C:\MesDocuments\DATASCIENTEST\PROJET\Livrable 3 Streamlit\CSV\\"
chemin_images = "C:\MesDocuments\DATASCIENTEST\PROJET\Livrable 3 Streamlit\IMAGES\\"
donnees_init_FULL = pd.read_csv(chemin_csv + 'DataFrame JDD_FULL.csv', sep=';', index_col=0).drop(["LUMINOSITE", "LUMINOSITE-CORR", "CONTRASTE", "CONTRASTE-CORR"], axis=1)
donnees_800_orig = pd.read_csv(chemin_csv + 'DataFrame JDD_800.csv', sep=';', index_col=0)    
donnees_800_corr = pd.read_csv(chemin_csv + 'DataFrame JDD_800_Cropped_LumCorr.csv', sep=';', index_col=0)    
data_simu = np.load(chemin_csv+'NP_800_Corr_data.npy') / 255
target_simu = to_categorical(np.load(chemin_csv+'NP_800_Corr_target.npy'), dtype = 'int')


#DEFINITION DES FONCTIONS
def AFFICHAGE_LUMIN_CONTRASTE(df, champ):
    #Cree un affichage violinplot d'un dataframe
    fig, axes = plt.subplots(1,1,  figsize=(5,5))
    sns.violinplot(ax=axes,x='TYPE', y=champ, data=df)
    return fig

def CALCULE_BORDS_BRUTS(masque):
    #Determine les nb de lignes/colonnes qui sont à 99% noirs dans une image
    nbPixels = masque.shape[0]
    xmin, xmax, ymin, ymax = (0,0,0,0)
    seuil = 1
    pctage = 0.99 * nbPixels
    for i in range(0,nbPixels//4):
        if (masque[:,i]<seuil).sum() > pctage: xmin = i
        if (masque[:,nbPixels-1-i]<seuil).sum() > pctage: xmax = i
        if (masque[i,:]<seuil).sum() > pctage: ymin = i
        if (masque[nbPixels-1-i,:]<seuil).sum() > pctage: ymax = i
    return xmin, xmax, ymin, ymax
    
def CALCULE_BORDS_CORRIGES(masque):
    #Determine les bords utiles d'une image, sous la forme d'un carré
    nbPixels = masque.shape[0]
    xmin, xmax, ymin, ymax = CALCULE_BORDS_BRUTS(masque)
    xmin_d, xmax_d, ymin_d, ymax_d = (0, 0, 0, 0)
    
    #Si les bords blancs sont déséquilibrés entre H et V, on se rabat sur la plus petite valeur afin de conserver les proportions de l'image initiale
    nbcrop = min(xmin + xmax, ymin + ymax)
    xmin_d = min(xmin, nbcrop)
    xmax_d = nbPixels - 1 - (nbcrop - xmin_d)
    ymin_d = min(ymin, nbcrop)
    ymax_d = nbPixels - 1 - (nbcrop - ymin_d)
    return xmin_d, xmax_d, ymin_d, ymax_d

def QUELLE_CLASSE_REELLE(num):
    #Détermine la classe réelle d'une image
    resu = target_simu[num] * [0, 1, 2, 3]
    return listeDesClasses[resu.sum()]
        
def QUELLE_CLASSE_PREDITE(test_pred, num):
    #Détermine la classe d'une image prédite par le modèle CNN
    resu = test_pred[num]
    resultat = ""
    for i in range(4):
        if resu[i]>0.1:
            if resultat != "":
                resultat += " + "
            resultat += str(int(resu[i]*100)) + "% " + listeDesClasses[i]
    return resultat


#DEBUT DE LA PRESENTATION
st.sidebar.title("Sommaire")

pages = ["Présentation",
         "Introduction", 
         "Analyse préalable", 
         "Gestion des biais",
         "Modélisation",
         "Démonstration",
         "Analyse",
         "Conclusions et perspectives"]
page = st.sidebar.radio("Aller vers", pages)


if page == pages[0]: #PRESENTATION
    st.title("Covid-19 pulmonary X-ray Analysis")
    st.write("Rapport de la formation **Data Scientist**")
    st.write("Soutenance le 7 octobre 2022")
    st.write("Présentateurs : Souad Bencharif, Lise Hong, Dan Touitou")
    st.write("https://github.com/BSouad/XRay-CovidPy")

elif page == pages[1]: #INTRODUCTION
    st.title(pages[1])
    st.image(Image.open(chemin_images + 'Lise_OIP_virus_structure.jpg'), caption="Structure moléculaire du virus")
    st.write("L'apparition du **SARS-CoV-2** (Severe Acute Respiratory Syndrome Coronavirus 2) en 2019 a créé une nouvelle maladie contagieuse de type respiratoire : la **COVID-19**.")
    st.image(Image.open(chemin_images + 'Lise_033020_eg_coronavirus-impact_feat.jpg'), caption="Répartition de la maladie dans le monde")
    
    st.write(" ")
    col1, col2 = st.columns(2)
    with col1:
        st.image(Image.open(chemin_images + 'Lise_OIP_symptomes.jpg'), caption="Symptômes liés à la maladie", use_column_width=True)
    with col2:
        st.write("La transmission du virus se fait principalement par inhalation de goutelettes du virus ou par contact de ces goutelettes avec les yeux, le nez et la bouche.")
        st.write("Les divers symptômes sont fièvre, toux, fatigue, difficultés respiratoires, perte d’odorat et de goût")
        st.write("* 1/3 de la population est asymptomatique")
        st.write("* 14% de la population a des symptômes sévères")
        st.write("* 5 % de la population développe des symptômes critiques")
    
    st.write(" ")
    st.write("La méthode la plus fiable actuellement pour le diagnostic est le **rRT-PCR** (real-time Reverse Transcription Polymerase Chain Reaction) mais non efficace rapidement.")
    st.write("Actuellement des recherches d’autres méthodes pour un  diagnostic plus rapide et à un stade plus précoce de la maladie sont en cours.")
    st.write("La radiographie pulmonaire est un bon candidat pour faire un diagnostic plus rapide. Cependant, c'est un travail fastidieux pour les radiologistes.")
    st.write("L'automatisation via des méthodes de type **Deep Learning** dans la prédiction de la maladie en utilisant les radiographies est une bonne solution. Cependant, les radiographies originales ne sont pas utilisables en raison de la présence de certains **biais** sur les images (artefacts, spécificité de certains appareils de radiographie, ...).")
    st.image(Image.open(chemin_images + 'Lise_Viral Pneumonia-1097.png'), caption="Exemple de biais.")
    st.write("Notre but est de corriger les biais pour un diagnostic des radiographies plus fiable.")


    st.title("Etat de l'art")
    st.write("Plusieurs axes de recherches :")
    st.write("* Préparation d’un jeu de données correct")
    st.write("* Détection des biais via les niveaux d’activation")
    st.write("* Quantification des biais via map attribution")
    st.write("* Exemple de Deep learning développé")
    st.write("* Utilisation du transfer learning pour prédire")
    st.write("* Exemple complet pour prédire la maladie")
    st.write("* Focus on XAI methods")
    st.write(" ")
    st.write("Selon les articles suivants")
    st.write(" 1. X. Yang, X. He, J. Zhao, Y. Zhang, S. Zhang, P Xie, 'COVID-CT-Dataset: A CT Scan Dataset about COVID-19', arXiv:2003.13865v3 , 2020")
    st.write("** https://arxiv.org/abs/2003.13865v3")
    st.write(" 2. I. Serna, A. Pena, A. Morales, J. Fierrez Julian, 'InsideBias: Measuring Bias in Deep Networks and Application to Face Gender Biometrics', arXiv:2004.06592V3, 2020")
    st.write("** https://arxiv.org/abs/2004.06592v3")
    st.write(" 3. N. Schaaf, O. De Mitri, B.K. Hang, A. Windberger, M.F.  Huber, 'Towards Measuring Bias in Image Classification', arXiv:2107.00360v1, 2021")
    st.write("** https://arxiv.org/abs/2107.00360v1")
    st.write(" 4. P. Afshar, S. Heidarian, N. Enshaei, F. Naderkhani, M. Rafiee Javad, OIKONOMOU A. Oikonomou, F. B. Fard, K. Samimi, K. N. Plataniotis, A. Mohammadia, 'COVID-CT-MD, COVID-19 computed tomography scan dataset applicable in machine learning and deep learning', Scientific Data. 2021; 8: 121., 2021")
    st.write("** https://europepmc.org/article/MED/33927208")
    st.write(" 5. S. Heidaran, AFSHAR P. Afshar, N. Enshaei, F. Naderkhani, RAFIEE M. J. Rafiee, F. B. Fard, K. Samimi, S F. Atashzar, A. Oikonomou, K. N. Plataniotis, A. Mohammadi 'COVID-FACT: A Fully-Automated Capsule Network-based Framework for Identification of COVID-19 Cases from Chest CT scans', Frontiers in Artificial Intelligence, 2021")
    st.write("** https://nyuscholars.nyu.edu/en/publications/covid-fact-a-fully-automated-capsule-network-based-framework-for-")
    st.write(" 6. W. Zhao, W. Jing, X. Qiu, 'Deep learning for COVID-19 detection based on CT images', Scientific Reports, 2021")
    st.write("** https://pesquisa.bvsalud.org/global-literature-on-novel-coronavirus-2019-ncov/resource/en/covidwho-1307346")
    st.write(" 7. M. Z. Islam, M. M. Islam,  A. Asraf, 'A combined deep CNN-LSTM network for the detection of novel coronavirus (COVID-19) using X-ray images', Elsevier Public Health Emergency Collection,  2020")
    st.write("** https://doaj.org/article/99b73b7dbd0b4a7080de7303050b3fce")
    st.write(" 8. I. Palatnik de Sousa,* M.M. B. R. Vellasco, E. Costa Da Silva, 'Explainable Artificial Intelligence for Bias Detection in COVID CT-Scan Classifiers', Sensors,2021")
    st.write("** https://doaj.org/article/78dfdab7ad6a42c48dbc7bb2fd2db64c")
    st.write(" 9. A. Haghanifar, Mahdiyar Molahasani Majdabadi, Y. Choi, D.S. Ko, 'COVID-CXNet: Detecting COVID-19 in Frontal Chest X-ray Images using Deep Learning', arXiv:2006.13807, 2020")
    st.write(" ** https://arxiv.org/abs/2006.13807")
    st.write(" 10 M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam,Can AI help in screening Viral and COVID-19 pneumonia? IEEE Access, Vol. 8, 2020, pp. 132665 - 132676.")
    st.write("** https://arxiv.org/abs/2003.13145")
    st.write(" 11 Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S.B.A., Islam, M.T., Maadeed, S.A., Zughaier, S.M., Khan, M.S. and Chowdhury, M.E., 2020.Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images.arXiv preprint arXiv:2012.02238.")
    st.write("** https://arxiv.org/abs/2012.02238")



elif page == pages[2]: #ANALYSE PREALABLE
    st.title(pages[2])
    st.header("Le dataset:")
    st.write("Le jeu de données est constitué de radiographies pulmonaires et de leurs masques, distribués selon 4 classes: COVID, Viral Pneumonia, Lung Opacity et Normal")
    st.write("Cette base de donnée a été constituée par une équipe de chercheurs de différentes universités et est disponible sur la plateforme Kaggle: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database")
    st.header("Construction d'un dataframe pandas")
    st.dataframe(donnees_init_FULL.head()) #add a title
    if st.checkbox("Afficher un exemple"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("<h6 style='text-align:center'>Exemple de radiographie</h6>", unsafe_allow_html=True)
            st.image(Image.open(chemin_images + 'Souad_img.png'))
        with col2:
            st.write("<h6 style='text-align:center'>Exemple de masque</h6>", unsafe_allow_html=True)
            st.image(Image.open(chemin_images + 'Souad_msk.png'))
    camembert = px.sunburst(donnees_init_FULL, path=["TYPE"])
    
    st.header("Mise en évidence de biais")
    if st.checkbox("Répartition des données"):
        st.write("<h6 style='text-align:center'>Distribution des données par classe</h6>", unsafe_allow_html=True)
        st.plotly_chart(camembert)
        st.write("* Sur-représentativité de la classe 'Normal'")
        st.write("* Sous-représentativité de la classe 'Viral Pneumonia")

    if st.checkbox("Luminosité et Contraste"):
    
        st.write("<h6 style='text-align:center'>Luminosité et contraste du dataset original</h6>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(AFFICHAGE_LUMIN_CONTRASTE(donnees_800_corr, "LUMINOSITE"))
        with col2:
            st.pyplot(AFFICHAGE_LUMIN_CONTRASTE(donnees_800_corr, "CONTRASTE"))
		
        st.write("* Luminosité moyenne moins importante pour les cas 'Normal'")
    
    if st.checkbox("Zones sans intérêt"):
        st.write("<h6 style='text-align:center'>Mise en évidence de zones noires sans intérêt particulier</h6>", unsafe_allow_html=True)
        st.image(Image.open(chemin_images + 'Souad_Borders.png'))
		
        st.write("* Les poumons n'occupent pas toujours la majeure partie de l'image, présence de bords noirs")
    
    if st.checkbox("Artefacts"):
        st.write("<h6 style='text-align:center'>Présence d'artefacts sur certaines images</h6>", unsafe_allow_html=True)
        st.image(Image.open(chemin_images + 'Souad_Artefacts.png'))
        
        st.write("* Présence d'annotations, pacemakers et autres artefacts sur certaines radiographies")


elif page == pages[3]: #GESTION DES BIAIS
    st.title(pages[3])
    types = donnees_init_FULL['TYPE'].value_counts()
    
    liste_biais = ["Uniformisation des classes",
                   "Utilisation des masques / Suppression des bords inutiles",
                   "Correction de la luminosité",
                   "Isomap avant et après corrections",
                   "Gradcam avant et après corrections"]

    st.header(liste_biais[0])
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(types)
    with col2:
        st.write("3 jeux de données :")
        st.write("- 4x200")
        st.write("- 4x500")
        st.write("- 4x1000")
    
    
    st.header(liste_biais[1])
    liste = ["Lung_Opacity-4662.png >> 8277",
             "COVID-1279.png >> 1278",
             "NORMAL-4699.png >> 14326",
             "Viral Pneumonia-728.png >> 20541"]
    
    image_choisie = st.selectbox(label = "Choisir une image", options = liste)
    num = int(image_choisie[-4:])
    img_orig = cv2.imread(donnees_init_FULL.FICHIER[num], cv2.IMREAD_GRAYSCALE)
    msk_orig = cv2.imread(donnees_init_FULL.MASQUE[num], cv2.IMREAD_GRAYSCALE)
    msk_299 = cv2.resize(msk_orig, (299,299), interpolation = cv2.INTER_AREA)
    xmin_d, xmax_d, ymin_d, ymax_d = CALCULE_BORDS_CORRIGES(msk_299)
    
    msk_bords = msk_299
    msk_bords[:, [xmin_d, xmax_d]] = 255
    msk_bords[[ymin_d, ymax_d]] = 255
    
    img_msk = cv2.add(img_orig, 255-msk_299)
    img_cropped = img_msk[ymin_d:ymax_d, xmin_d:xmax_d]
    img_resized = cv2.resize(img_cropped, (299,299), interpolation = cv2.INTER_AREA)
    
    image_options = ["Images originales",
                     "Masques originaux",
                     "Identification des bords inutiles",
                     "Fusion de l'image et du masque",
                     "Redimensionnement"]
    choix = st.select_slider("Déroulement des étapes...", options=image_options)

    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(img_orig)
    with col2:
        if (choix == image_options[1] or choix == image_options[2] or choix == image_options[3] or choix == image_options[4]):
            st.image(msk_orig)
    with col3:
        if (choix == image_options[2] or choix == image_options[3] or choix == image_options[4]):
            st.image(msk_bords)
    with col4:
        if (choix == image_options[3] or choix == image_options[4]):
            st.image(img_msk)
    with col5:
        if choix == image_options[4]:
            st.image(img_resized)

    
    st.header(liste_biais[2])
    st.subheader("Pour chaque pixel : X = X - X[TYPE].mean() + X.mean()")
    

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("<h6 style='text-align:center'>JDD initial</h6>", unsafe_allow_html=True)
        st.pyplot(AFFICHAGE_LUMIN_CONTRASTE(donnees_800_corr, "LUMINOSITE"))
    with col2:
        st.write("<h6 style='text-align:center'>JDD + cropping</h6>", unsafe_allow_html=True)
        st.pyplot(AFFICHAGE_LUMIN_CONTRASTE(donnees_800_corr, "LUMINOSITE-CORR"))
    with col3:
        st.write("<h6 style='text-align:center'>JDD + cropping + correction lum</h6>", unsafe_allow_html=True)
        st.pyplot(AFFICHAGE_LUMIN_CONTRASTE(donnees_800_corr, "LUMINOSITE-CORR2"))


    st.header(liste_biais[3])
    col1, col2 = st.columns(2)
    with col1:
        st.write("<h6 style='text-align:center'>Isomap sur JDD_4000 initial</h6>", unsafe_allow_html=True)
        st.image(Image.open(chemin_images + "Isomap_4000_orig.png"))
    with col2:
        st.write("<h6 style='text-align:center'>Isomap sur JDD_4000 corrigé</h6>", unsafe_allow_html=True)
        st.image(Image.open(chemin_images + "Isomap_4000_corr.png"))

    st.header(liste_biais[4])
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write("<h6 style='text-align:center'>COVID-1492</h6>", unsafe_allow_html=True)
        st.image(Image.open(chemin_images + "Gradcam_COVID-1490.png"))
    with col2:
        st.write("<h6 style='text-align:center'>Lung_Opacity-4013</h6>", unsafe_allow_html=True)
        st.image(Image.open(chemin_images + "Gradcam_Lung_Opacity-4013.png"))
    with col3:
        st.write("<h6 style='text-align:center'>Normal-4736</h6>", unsafe_allow_html=True)
        st.image(Image.open(chemin_images + "Gradcam_Normal-4736.png"))
    with col4:
        st.write("<h6 style='text-align:center'>Viral Pneumonia-71</h6>", unsafe_allow_html=True)
        st.image(Image.open(chemin_images + "Gradcam_Viral Pneumonia-71.png"))


elif page == pages[4]: #MODELISATION
    st.title(pages[4])
    st.write("**Etapes préalables :**")
    st.write("- Conversion des JDD en Numpy Arrays")
    st.write("- Split des JDD en jeux d'entrainement (80%) et de test (20%)")
    
    num_pixels = 299
    num_classes = 4
    
    #CONSTRUCTION D'UN MODELE CNN
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation='relu', input_shape=[299, 299, 1]))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    model.add(tf.keras.layers.Dropout(rate = 0.2))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    model.add(tf.keras.layers.Dropout(rate = 0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate = 0.2))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    
    st.write("**Présentation du modèle :**")
    if st.checkbox("Modèle CNN théorique"):
        st.image(Image.open(chemin_images + "Modele theorique CNN.jpg"))
    if st.checkbox("Résumé du modèle sélectionné"):
        model.summary(print_fn=lambda x: st.text(x))

    st.write("**Exécution du modèle :**")
    
    jdd = st.select_slider('Choix du jeu de données',
                           options=("800_Orig", "800_Corr", "2000_Orig", "2000_Corr", "4000_Orig", "4000_Corr"))
    nb_epochs = st.slider('Nombre d epochs', 2, 25, 2)
    bsize = st.select_slider('Taille du batch', options=(8, 16, 32, 48, 64))
    
    if st.button('Entrainement du modèle'):
        st.session_state.bouton_train = "Pendant"
    if st.session_state.bouton_train == "Pendant":
        start_time = datetime.now()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        #CHARGEMENT DES NUMPY ARRAY
        data = np.load(chemin_csv+'NP_' + str(jdd) + '_data.npy')
        target = np.load(chemin_csv+'NP_' + str(jdd) + '_target.npy')
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
        X_train = X_train / 255
        X_test = X_test / 255
        y_train = to_categorical(y_train, dtype = 'int')
        y_test = to_categorical(y_test, dtype = 'int')
        
        st.session_state.training_history = model.fit(X_train, y_train, epochs=nb_epochs, batch_size=bsize, validation_split = 0.2)
        end_time = datetime.now()
        st.write('Duree d execution : ', (end_time - start_time).total_seconds(), ' secondes');
        st.session_state.bouton_train = "Après"
    if st.session_state.bouton_train == "Après":
        fig = plt.figure(figsize=(20,6))
        #nb_epochs = 2
        # Courbe de la précision sur l'échantillon d'entrainement / de test
        ax1 = fig.add_subplot(121)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.plot(np.arange(1 , nb_epochs+1, 1), st.session_state.training_history.history['accuracy'], "r--", label = 'Training Accuracy')
        ax1.plot(np.arange(1 , nb_epochs+1, 1), st.session_state.training_history.history['val_accuracy'], "r", label = 'Validation Accuracy')
        ax1.legend()
        # Courbe de perte sur l'échantillon d'entrainement / de test
        ax2 = fig.add_subplot(122)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.plot(np.arange(1 , nb_epochs+1, 1), st.session_state.training_history.history['loss'], "b--", label = 'Training Loss')
        ax2.plot(np.arange(1 , nb_epochs+1, 1), st.session_state.training_history.history['val_loss'], "b", label = 'Validation Loss')
        ax2.legend()
        st.pyplot(fig)

    if st.checkbox("Courbes d'accuracy et de perte pour plusieurs simulations :"):
        st.write("<h6 style='text-align:center'>JDD 800_Corr, 25 epochs, batch_size=16 (15 min)</h6>", unsafe_allow_html=True)
        st.image(Image.open(chemin_images + "Loss_800_Corr_epoch25_batch16.png"))
        st.write("<h6 style='text-align:center'>JDD 2000_Corr, 25 epochs, batch_size=16 (30 min)</h6>", unsafe_allow_html=True)
        st.image(Image.open(chemin_images + "Loss_2000_Corr_epoch25_batch16.png"))
        st.write("<h6 style='text-align:center'>JDD 4000_Corr, 25 epochs, batch_size=16 (60 min)</h6>", unsafe_allow_html=True)
        st.image(Image.open(chemin_images + "Loss_4000_Corr_epoch25_batch16.png"))


elif page == pages[5]: #DEMONSTRATION
    st.title(pages[5])
    
    if st.checkbox("Matrice de confusion pour le JDD 800, avec 25 epochs et batch_size=16"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("<h6 style='text-align:center'>Images originales</h6>", unsafe_allow_html=True)
            st.image(Image.open(chemin_images + "Confusion_800_Orig_epoch25_batch16.png"))
        with col2:
            st.write("<h6 style='text-align:center'>Images corrigées</h6>", unsafe_allow_html=True)
            st.image(Image.open(chemin_images + "Confusion_800_Corr_epoch25_batch16.png"))
        st.write("Legend : 0 = COVID, 1 = Lung_Opacity, 2 = Normal, 3 = Viral Pneumonia")


    if st.checkbox("Prédiction de la classe d'une image par le modèle CNN"):
        image_choisie = st.selectbox(label = "Choisir l'image qui sera soumise au modèle CNN afin qu il prédise sa classe", 
                                     options = [x for x in range(800)])
        col1, col2 = st.columns(2)
        with col1:
            st.image(Image.open(donnees_800_orig.FICHIER[donnees_800_orig.index[image_choisie]]))
        with col2:
            st.image(Image.open(donnees_800_corr.FICHIER[donnees_800_corr.index[image_choisie]]))
        
        listeDesClasses = ["COVID", "Lung_Opacity", "NORMAL", "Viral Pneumonia"]
        
        if st.session_state.chargement_model == False:
            st.write("Chargement du modèle entrainé (10-15 sec) ...")
            model_orig = tf.keras.models.load_model("Model_800_Corr_25ep_16btch.h5")
            st.session_state.test_pred_orig = model_orig.predict(data_simu)
            model_corr = tf.keras.models.load_model("Model_800_Orig_25ep_16btch.h5")
            st.session_state.test_pred_corr = model_corr.predict(data_simu)
            st.session_state.chargement_model = True
        
        st.write("Classe réelle de l'image : " + QUELLE_CLASSE_REELLE(image_choisie))
        st.write("Prédictions du modèle CNN pour l'image originale : " + QUELLE_CLASSE_PREDITE(st.session_state.test_pred_orig, image_choisie))
        st.write("Prédictions du modèle CNN pour l'image corrigée : " + QUELLE_CLASSE_PREDITE(st.session_state.test_pred_corr, image_choisie))
        
        #st.write("Images intéressantes : num / classe réelle / prédiction Orig / prédiction Corr")
        #for inum in range(800):
        #    st.write(str(inum) + " : " + QUELLE_CLASSE_REELLE(inum) + " / " + QUELLE_CLASSE_PREDITE(st.session_state.test_pred_orig, inum) + " / " + QUELLE_CLASSE_PREDITE(st.session_state.test_pred_corr, inum))
        
        
elif page == pages[6]: #ANALYSE
    st.title(pages[6])
    st.write("**Tableau récapitulatif des résultats**")
    st.image(Image.open(chemin_images + 'Lise_results.png'), 
             caption="Tableau de synthèse sur les 4 jeux de données testés. Les valeurs vertes sont les meilleures et les oranges les moins bonnes.")
    st.write("Sur le training du modèle CNN sur tout l’ensemble du dataset, la matrice de confusion indique que la prédiction s’est effectuée avec un score assez élevé.")
    st.write("Le modèle semble performant mais la prédiction n’est pas focalisée sur l’infection (ce sont les biais qui augmentent son score.")
    st.write("Ceci est prouvé par le jeu de données sur les images segmentées seulement (plus faible score).")
    st.write("L'entrainement sur un jeu de 800 images (sur 4 simulation différentes) ne permet pas de faire une bonne prédiction, que le preprocessing soit effectué ou non.")
    st.write("Seule la classe 4 est mieux prédite (Viral pneumonia) pour ces jeux de données.")
    st.write("Un ou plusieurs critères sont des caractéristiques spécifiques à l’infection qui sont reconnus par notre modèle.")
    st.write(" ")
    st.write("Au cours de la rédaction de la soutenance, d'autres preprocessings et entraînements sur un modèle différent ont été faits et les résultats sont beaucoup plus mitigés.")
    st.write("En effet, les entraînements sur 800, 2000 et 4000 images ont été faits et les courbes de loss indiquent que quelque soit le modèle entraîné, notre modèle ne se généralise pas.")
    st.write("La prédiction est d'ailleurs meilleure sur les images originales.")


elif page == pages[7]: #CONCLUSIONS
    st.title("Conclusions")
    st.write("Ce projet s’inscrit dans le cadre d’un sujet d’actualité.")
    st.write("Plusieurs pistes de recherche ont été suivies dont la détection de la COVID-19 avec le deep learning.")
    st.write("Après analyse de travaux réalisés, la piste de la correction du biais semble peu explorée.")
    st.write("Notre travail portant sur cette correction comporte plusieurs étapes")
    st.write("préprocessing pour corriger le biais")
    st.write("entraînement sur un modèle qui semble être plus approprié")
    st.write("La correction des biais ne semble pas apporter une meilleure prédiction de la maladie via les preprocessing et entraînements effectués mais d'autres tests sont nécessaires pour confirmer ou infirmer cette hypothèse.")

    st.title("Perspectives")
    st.write("L’entraînement sur un jeu de 800 images semble être insuffisant. Il faut aussi augmenter le nombre d’images dans le jeu.")
    st.write("L’entraînement sur 25 epoch et un modèle standard de 5 couches : changement de ces paramètres.")
    st.write("L’entraînement ne s’est fait qu’une fois. Rejouer plusieurs fois sur de différents échantillons homogénéisées peut être une alternative pour voir l’augmentation.")
    st.write("Utiliser d’autres techniques comme l’augmentation, le cropping ...")