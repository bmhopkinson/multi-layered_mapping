# pycamgeom
Package for projective camera and associated classes

Camera: projective camera class
Frame: represents a image and its pose in world coordinates
AABBtree: used to accelerate searches for world entities visible in camera image
Projector: projects faces and vertices from mesh into frame to identify visible mesh elements


# for uploading to pypi:
1. python3 -m build from root dir (/pycamgeom)
2. python3 -m twine upload --repository pypi dist/*

# for local install (development testing)- couldnt' get pip install -e . option to work 
1. pip uninstall pycamgeom 
2. pip install . (from root: ie. /pycamgeom)