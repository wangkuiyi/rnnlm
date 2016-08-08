# Using Bazel

As we are trying to train the model using Tensorflow, it is mandatory
that I link my decoder with Tensorflow Serving C++ libraries.  Since
Tensorflow and Tensorflow Serving are built from source code using
Bazel, I am learning Bazel.

## Bazel in Docker

### Build the Docker Image

I program on Mac, but I want my code built for Linux, so I use Docker.
The Dockerfile in this directory builds a Docker image from Ubuntu
14.04 and with Bazel installed.

To build the image, run

```
docker build -t bazel -f bazel.Dockerfile .
```

### Build in a Docker Container

I have some example projects that are buildable using Bazel in
`./examples`.  The simplest one is `./examples/cpp`.  To build it
using Bazel Docker container, run

```
docker run --rm \
-v /Users/yiwang/work/rnnlm:/rnnlm \
-v /tmp:/tmp \
-it bazel \
/bin/bash -c \
"cd /rnnlm/bazel/examples && bazel --output_base /tmp build //cpp:hello"
```

Note that `-v /Users/yiwang/work/rnnlm:/rnnlm` mounts local source
directory to `/rnnlm` in the Docker container.  `-v /tmp:/tmp` and
`bazel --output_base /tmp` jointly make sure that the generated files
are placed in `/tmp` of my Mac, instead of `/tmp` in the container.

In this way, I program on my Mac using Emacs and build using Bazel in
Docker.

Note that you might want to add the following line to your `~/.emacs`
file in you use Emacs with Bazel, so that Emacs will highlight `BUILD`
files with Python syntax.

```
(add-to-list 'auto-mode-alist '("\\BUILD\\'" . python-mode))
```

### Run in a Docker Container

After the building, we can run the generated binary file,
`examples/bazel-bin/cpp/hello`, in another Docker container:

```
docker run --rm \
-v $(pwd)/examples/bazel-bin/cpp:/tmp \
-it ubuntu:14.04 /tmp/hello
```

## External Projects

Each directory containing a `WORKSPACE` file is considered a project
by Bazel.  For example, `https://github.com/tensorflow/serving` is a
project.  However, this project refers to another Bazel project
`https://github.com/tensorflow/tensorflow`.  For the former to depend
itself on the latter:

1. We need to declare the latter in the former's `WORKSPACE` file, and
1. we need to declare some rules in the latter's BUILD file as
   *visible* by other projects.

For example, we created `./another_example/hello_world:hello_world`,
which depends on `./examples/cpp:hello-lib`.  Here
`./another_example/WORKSPACE` denotes `./another_example` a new Bazel
project.  We can see that:

1. In `./another_example/WORKSPACE`, we use `local_repository` to
   declared `./examples` as an *external project*, and
1. in `./examples/cpp/BUILD`, we use `visibility =
   ["//visibility:public"],` to denote rule `hello-lib` a public one,
   so it can be referred to by other projects.

To build `./another_example/hello_world`, run:

```
docker run --rm \
-v /Users/yiwang/work/rnnlm:/rnnlm \
-v /tmp:/tmp \
-it bazel \
/bin/bash -c "cd /rnnlm/bazel/another_example && bazel --output_base /tmp build //hello_world:hello_world"
```

To run the build `hello_world` binary file, run

```
docker run --rm \
-v $(pwd)/another_example/bazel-bin/hello_world:/tmp \
-it ubuntu:14.04 \
/tmp/hello_world
```

## Troubleshooting

1. Bazel use directory names as package names, and it requires that
   package names consist of `A-Z`, `a-z` and `_`, so directory names
   like `c++` doesn't work with Bazel.
