[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gouarin/formation_lycee_2020/master)

Pour utiliser et modifier les notebooks chez vous, il vous suffit de suivre la procédure suivante:

1- Installer miniconda (https://docs.conda.io/en/latest/miniconda.html)

2- Télécharger ce dépôt (https://github.com/gouarin/formation_lycee_2020/archive/master.zip)

3- Décompresser l'archive

4- Ouvrir un Anaconda prompt (https://docs.anaconda.com/anaconda/user-guide/getting-started/#open-anaconda-prompt)

5- Aller dans le répertoire et exécuter les commandes suivantes

```
conda env create -f binder/environment.yml
conda activate enseignement_2020
jupyter notebook
```

Vous devriez avoir une page de votre navigateur qui s'ouvre et voir le répertoire des notebooks.