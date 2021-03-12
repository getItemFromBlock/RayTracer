# RayTracer
Version réécrite en langage C++/CUDA du projet Ray-Tracing

ChangeLog

12/03/2021

Ajout des fichiers manquants "SphereDrawable", "TriangleDrawable" et "VoidDrawable"

Ajout de l'objet "TriangleMirrorDrawable", qui est la version réfléchissante du triangle

Mise à jour des fichiers "Client" et "ObjectHolder":

-> Support pour ajouter l'objet TriangleMirrorDrawable

-> Les objets ajoutés depuis le script client on maintenant des fonction prédéfinies

-> Support pour le futur objet SphereMirrorDrawable

-> modification de la fonction hit dans "ObjectHolder"
