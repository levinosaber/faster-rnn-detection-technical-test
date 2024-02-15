# Environnement
normalement, les environnements utilisé n＇est pas compliqué mais pour être sûr que le programme marche bien，j＇ai décidé de faire freeze tous les dépendances de mon environnement virtuel dans le fichier txt

# utilisation 
Après avoir mis le jeu de données dans le même répertoire de ce projet et le nommé "dataset",
Exécuter le main_pipeline.py, tout sera fait automatiquement dans la condition respectant l'existence de répertoire "runs" pour stocker les résultats d'une exécution
Chaque fois, il faut être sûr qu'il n'a pas de répertoire avec le même nom que training_opts["splitted_dataset_path"] défini dans main_pipeline.py.
Tous les paramètres qui peut être modifié sont indiqué dans lès commentaires dans main_pipeline.py
