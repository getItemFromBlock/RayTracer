# RayTracer

## Version réécrite en langage C++/CUDA du projet Ray-Tracing



## ChangeLog

### 10/10/2021

Réorganisation des fichiers
Ajout du nouveau type "SphereDrawableMirror"
Debug du système de lumière et d'ombres
Ajout de shading pour les surfaces éclairées:non éclairées
Ajout de collisions
Ajout des métadata, premettant de désactiver certains paramètres sur les objets:
  -  Désactiver les collisions
  -  Désactiver la projection d'ombre
Appuyer sur 'w' permet de zoomer
Optimisations globales


### 12/03/2021

Ajout des fichiers manquants "SphereDrawable", "TriangleDrawable" et "VoidDrawable"

Ajout de l'objet "TriangleMirrorDrawable", qui est la version réfléchissante du triangle

Mise à jour des fichiers "Client" et "ObjectHolder":

-> Support pour ajouter l'objet TriangleMirrorDrawable

-> Les objets ajoutés depuis le script client on maintenant des fonction prédéfinies

-> Support pour le futur objet SphereMirrorDrawable

-> modification de la fonction hit dans "ObjectHolder"
