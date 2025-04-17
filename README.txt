**BoT-SORT + YOLO - Suivi multi-objets de chiens à partie de cameras traps**

Description :
--------------
Ce programme applique un algorithme de **tracking multi-objets** basé sur **BoT-SORT**, combiné à un modèle de détection **YOLO** (You Only Look Once), sur une vidéo fournie.  

Il permet de détecter uniquement les **chiens** présents à l’image (grâce à un poids YOLO personnalisé) et de leur assigner une identité unique tout au long de la séquence vidéo.

Ceci est la seconde version (v2.0) du programme,

**Modifications / Ajouts notables :**
- Compatible avec un modèle YOLOv11 et sélectionnant uniquement la classe 'dog' de COCO.
- Affichage du seuil de confiance sur les frames

--------------
Auteur : Antoine Lebourg  
Stage LIRMM 2025 – Projet SEAdogSEA  
Dernière mise à jour : Mai 2025