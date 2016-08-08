# Using Bazel

As we are trying to train the model using Tensorflow, it is mandatory
that I link my decoder with Tensorflow Serving C++ libraries.  Since
Tensorflow and Tensorflow Serving are built from source code using
Bazel, I am learning Bazel.

## Bazel in Docker

I program on Mac, but I want my code built for Linux, so I use Docker.
The Dockerfile in this directory builds a Docker image from Ubuntu
14.04 and with Bazel installed.

To build the image, run

```
docker build -t bazel -f bazel.Dockerfile .
```

## Docker Examples

I have some example projects that are buildable using Bazel in
`./examples`.  The simplest one is `./examples/cpp`.  To build it
using Bazel Docker container, run

```
docker run --rm -v /Users/yiwang/work/rnnlm:/rnnlm -v /tmp:/tmp -it bazel /bin/bash -c "cd /rnnlm/bazel && bazel --output_base /tmp/bazel build //examples/cpp:hello-lib"
```

Note that `-v /Users/yiwang/work/rnnlm:/rnnlm` mounts local source
directory to `/rnnlm` in the Docker container.  `-v /tmp:/tmp` and
`bazel --output_base /tmp` jointly make sure that the generated files
are placed in `/tmp` of my Mac, instead of `/tmp` in the container.


## Troubleshooting

1. Bazel use directory names as package names, and it requires that
   package names consist of `A-Z`, `a-z` and `_`, so directory names
   like `c++` doesn't work with Bazel.
