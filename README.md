## Multi-Layered Mapping

Tools to transfer information (semantic segmentation, detected objects) onto a triangular mesh from registered images. As illustrated in the diagram below if the pose of an image and its associated camera model is known with respect to a triangular mesh surface representations, information from the images can be back-projected onto the mesh. Image information can include class labels, objects, etc and is typically produced automatically using trained machine learning models.

<img src=doc/box_1_diagram.jpg width="500">

An example of a multi-layered map of a salt marsh is shown below. The map included plant class information (colors) and locations of individual snails (grey dots).
![example](doc/marsh_example.jpg)

## Modules
1. **pycamgeom**: python package for camera models and data structures (AABB hierarchy) for accelarting camera projection/backprojection operations.
2. **mesh\_class_labeling**: transfers class information from semantically segmented images onto triangular mesh
3. **mesh\_object_placing**: backprojects objects detected in images onto triangular mesh.

