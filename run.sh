docker build -t ocamlmycaml/w266-final-project:latest .
docker run --runtime=nvidia \
    -it -p 8888:8888 -p 6006:6006 --rm \
    -u $(id -u):$(id -g) \
    -v "$(pwd)":/tf/notebooks \
    ocamlmycaml/w266-final-project:latest
