#!/bin/bash
# Use the arm64 docker image in https://hub.docker.com/r/multiarch/debian-debootstrap/ for unit tests in Travis CI

sudo docker run --rm --privileged multiarch/qemu-user-static:register --reset
sudo docker run -v $(pwd):/farm --rm multiarch/debian-debootstrap:arm64-jessie bash -c \
    "apt-get -y update; apt-get -y install build-essential; cd /farm/test; make test; ./bin/test_correctness"
