#!/bin/sh

git clone https://github.com/paulbeka/phylogenetic_tree_prediction.git

curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

cd code

./docker run -v /output:/app/src/output phylogenetic_data