"""Le corps numérique de Valdar : son environnement aujourd'hui, c'est le PC d'Olivier.

Il le voit (capture d'écran, fenêtres et boutons lus par l'accessibilité de Windows) et il
peut y agir (fichiers, terminal, souris, clavier, applications), avec des garde-fous :
- réservé au propriétaire reconnu ;
- des zones interdites (secrets, mots de passe des navigateurs, empreintes biométriques) ;
- son propre code et ses garde-fous ne sont jamais modifiables par lui ;
- tout ce qui est risqué (écrire hors de son atelier, commande qui modifie, clic sur
  « Supprimer » ou « Envoyer », Entrée après une saisie…) demande un « oui » ;
- jamais de saisie dans un champ de mot de passe ;
- chaque action est notée dans data/corps/journal.jsonl.

Un corps robotique est prévu plus tard : rien ici ne le suppose.
"""
