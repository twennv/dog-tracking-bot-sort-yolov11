BoT-SORT + YOLOv11 - Suivi multi-objets de chiens à partir de cameras traps

--------------

Description :

Ce programme applique un algorithme de tracking multi-objets basé sur BoT-SORT, combiné à un modèle de détection YOLOv11, sur une vidéo fournie.  

Il permet de détecter uniquement les chiens présents à l’image (modèle YOLOv11 utilisant uniquement la classe 'dog') et de leur assigner une identité unique tout au long de la séquence vidéo.

Ceci est la seconde version (v2.1) du programme,

Modifications / Ajouts notables :
- Compatible avec un modèle YOLOv11 et sélectionnant uniquement la classe 'dog' de COCO.
- Affichage du seuil de confiance sur les frames
- Affichage dynamique
- Optimisation global du code

--------------

Version Google Colab :

- Version v1.0 (BoT-SORT + Yolov5 fine tuné)
https://colab.research.google.com/drive/1DB3Dz4FFHl7LP5jlxm703Yc65wB0Enc5?authuser=1#scrollTo=SdUSf__029_0

- Version v2.0 (BoT-SORT + Yolov11, moins optimisé -> code + hyperparamètre à revoir)
https://colab.research.google.com/drive/1DB3Dz4FFHl7LP5jlxm703Yc65wB0Enc5?authuser=1#scrollTo=SdUSf__029_0

Il est conseillé d'utiliser la version 2.1 locale de cette solution de tracking.

--------------

Auteur : Antoine Lebourg
Stage LIRMM 2025 - Projet SEAdogSEA  
Dernière mise à jour : Mai 2025
