# Foreword
This document is a description of my/Antoine Webanck's submission to the Shadertoy contest of the JFIG2021.
The shader is entitled `GrasseJasmin` and its link is https://www.shadertoy.com/view/NldGR8.

## Global description
A flask of perfume on a table top, accompanied by a jasmin flower tip.
The flask is labelled as this edition of the JFIG.

## Inspiration
Edition 2021 of the JFIG takes place on the campus of Sophia Antipolis.
Sophia is located in the region Provence-Alpes-Côte d'Azur which also contains the town of Grasse.
Grasse is a town well known for its production of perfume, and notably the culture of Jasmin.
The particular species cultivated in Grasse is the *Jasminum grandiflorum*.

A flask of perfume is a noble manufactured object, a symbol of luxury often crystalized by its design.
The common blend of glass, gold, and the liquid in a geometric or complex shape allows for mesmerizing and fascinating interplays with light.
Even the simple shape of the flask of the shader creates interesting reflection and refraction patterns through glass interface.

A flower is a complex and fragile reroduction organ produced by plants to attract pollinating insects thanks to its perfume, shape and colors.
Flowers are beautiful complex, organic and intricate shapes and colors.
As such, a flower is a challenging object to model and even more to animate.

## Technical details
The objects of the scene are modeled with distance functions to primitives, boolean operators and a little bit of domain warping.
The scene is rendered by path tracing and ray-object intersections are computed by ray-marching the global distance function.
At each intersection, the normal is computed as the gradient of the distance function.

In bulk:
- Diffuse surfaces, glass refraction/reflection.
- The JFIG label is adapted from the bitfield from the previous year (see https://github.com/ssloy/tinyraytracer/wiki/Part-3:-shadertoy).
- Buffered rendering for incremental rendering.

# Acknowledgment/bibliography
Thanks to Inigo Quilez as always, notably for his work and ressources on distance functions.

