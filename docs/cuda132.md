## How can we install cuda 13.2?

1. First install cuda 13.2 (linux only, nvidia-driver-595 with open needed)
2. install fa2 and build. I tried build in python3.13, prebuilt [wheel](https://github.com/sorryhyun/flash-attention-sm120-fix/releases/download/fa2cuda132/flash_attn-2.8.3-cp313-cp313-linux_x86_64.whl)
3. install bitsandbytes from source and build with cmake


## How faster it is?

* In some extent, like ~10%