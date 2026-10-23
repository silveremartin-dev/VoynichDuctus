Et si l'on étudiait le manuscrit de Voynich à partir de l'encre, et non de nos transcriptions ?
Je publie VoynichDuctus, un pipeline open source de paléographie numérique. 🖋️

Depuis les années 1990, la plupart des études computationnelles du manuscrit (Beinecke MS 408, daté 1404–1438) portent sur l'alphabet EVA, une transcription humaine. Elles mesurent donc en partie nos propres choix de découpage des glyphes (ligatures, allographes), et pas seulement le geste du scribe.

L'approche : repartir des scans haute résolution, reconstruire le ductus (ordre et direction des traits, largeur de plume) puis laisser émerger un alphabet par clustering non supervisé, sans étiquettes humaines.

Ce que contient le dépôt :
• Binarisation adaptative (Sauvola, Wolf-Jolion) et recalage SIFT/RANSAC sur les 225 folios Yale
• Vectorisation par modèle physique de plume biseautée à 40° et squelette Euler-Bernoulli, comparés à DINOv2 et à un adaptateur InkSight
• Clustering en archétypes (G01…G35) : ~200 000 glyphes Voynich, 5 258 pour le Codex Seraphinianus
• Diagnostics : entropies H1/H2, exposant de Hurst (DFA), compressibilité LZMA, déterminisme markovien, face à un herbier latin du XVe s. et aux générateurs de Timm et de Rugg
• Un explorateur HTML interactif et 52 tests

📊 Premier résultat : sur 1 387 tokens, les tokens issus du ductus présentent un déterminisme markovien d'ordre 2 plus élevé (70,5 %) que la transcription EVA (50,5 %).

⚠️ Limites : aucune traduction ni « déchiffrement ». Les échantillons sont petits, et le classifieur de diagnostic est encore fragile (il range le texte latin témoin parmi les textes « auto-citation de Timm »). Il s'agit d'une base de travail reproductible, pas d'une conclusion.

Retours de paléographes, linguistes et spécialistes ML bienvenus, ainsi que les contributions :
👉 https://github.com/silveremartin-dev/VoynichDuctus

#PaléographieNumérique #HumanitésNumériques #Voynich #VisionParOrdinateur #OpenSource #MachineLearning
