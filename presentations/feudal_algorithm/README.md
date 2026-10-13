# Feudal MAPPO presentation

Eight slides, approximately 8–10 minutes. White background, restrained blue accent,
editable PowerPoint text and diagrams, rendered equations, and speaker notes.

- `feudal_algorithm.pptx`: presentation with embedded speaker notes.
- `feudal_algorithm.pdf`: matching PDF for sharing.
- `speaker_notes.md`: explanations, all equation terms, source references, and LaTeX.
- `build_short_deck.py`: source for the delivered eight-slide deck.
- `build_slides.py`: preserved, earlier long-draft source; not used for the delivered deck.

To rebuild, install `python-pptx`, `reportlab`, `matplotlib`, and `Pillow`, and run
`python build_short_deck.py`. The renderer expects DejaVu Sans in the standard
Linux font directory. Equations are embedded at 240 dpi; edit their LaTeX in the
source to regenerate them. Text and diagrams remain native PowerPoint objects.

Sources are the working-tree implementation under `algorithms/feudal_mappo_jax/`
and the defaults under `conf/algorithm/feudal_mappo_jax.yaml`, inspected on
24 September 2026. Algorithm behavior is explained, not benchmarked.
