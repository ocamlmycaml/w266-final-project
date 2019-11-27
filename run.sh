docker build -t ocamlmycaml/tf-models:latest .
docker run --runtime=nvidia \
    -it -p 8888:8888 --rm \
    -u $(id -u):$(id -g) \
    -v "$(pwd)":/tf/notebooks \
    ocamlmycaml/tf-models:latest
