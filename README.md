# Music Generation

à partir des midi, on va generer une musique.

## Décomposition en série de fourier

- [ ] Chaque .mid -> lecture -> extraire la frequence, amplitude ( serie de fourier )
- [ ] Modèle qui décompose une musique en série d'instrument.
- [ ] Preprocessing ( suppression voix, conserver 3 instruments, 1 ou 2 genres )
- [ ] Modelisation: modele de génération -> Convolutional Neural Networks ( CNN ) ( 2 - 3 layers )

## Dataframe

[[instrument1:Frequence1]...[instrumentN:FrequenceN]]

# Music Reconnaissance

à partir des midi, on va chercher à trouver le genre de la musique 

## Décomposition en série de fourier

- [ ] Preprocessing ( suppression voix, conserver 3 instruments, tout les genres ( selection ROCK, RAP, CLASSIQUE, POP) )

## Dataframe

[[instrument1:Frequence1]...[instrumentN:FrequenceN]] -> [genre] ( classification )
