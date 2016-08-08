cc_library(
    name = "hello-lib",
    srcs = ["hello-lib.cc"],
    hdrs = ["hello-lib.h"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "hello",
    srcs = ["hello.cc"],
    deps = [
        ":hello-lib",
    ],
)
