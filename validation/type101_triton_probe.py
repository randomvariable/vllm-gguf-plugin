import torch
import triton
import triton.language as tl


@triton.jit
def codebook(code):
    return tl.where(
        code == 0,
        0.0,
        tl.where(
            code == 1,
            1.0,
            tl.where(
                code == 2,
                2.0,
                tl.where(
                    code == 3,
                    3.0,
                    tl.where(
                        code == 4,
                        4.0,
                        tl.where(
                            code == 5,
                            6.0,
                            tl.where(
                                code == 6,
                                8.0,
                                tl.where(
                                    code == 7,
                                    10.0,
                                    tl.where(
                                        code == 8,
                                        0.0,
                                        tl.where(
                                            code == 9,
                                            -1.0,
                                            tl.where(
                                                code == 10,
                                                -2.0,
                                                tl.where(
                                                    code == 11,
                                                    -3.0,
                                                    tl.where(
                                                        code == 12,
                                                        -4.0,
                                                        tl.where(
                                                            code == 13,
                                                            -6.0,
                                                            tl.where(
                                                                code == 14,
                                                                -8.0,
                                                                -10.0,
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


@triton.jit
def probe(decoded_ptr, result_ptr, weights_ptr, mode: tl.constexpr):
    offs_m = tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    offs_k = tl.arange(0, 16)
    packed = tl.load(weights_ptr + offs_n[:, None] * 17 + offs_k[None, :])
    low = packed & 0xF
    high = packed >> 4

    scale_byte = tl.load(weights_ptr + offs_n * 17 + 16)
    exponent = (scale_byte >> 3) & 0xF
    mantissa = scale_byte & 7
    scale = tl.where(
        scale_byte > 0x7E,
        0.0,
        tl.where(
            exponent == 0,
            mantissa.to(tl.float32) / 1024.0,
            (1.0 + mantissa.to(tl.float32) / 8.0)
            * tl.exp2(exponent.to(tl.float32) - 8.0),
        ),
    )

    if mode == 0:
        w_tile = tl.join(codebook(low), codebook(high))
    elif mode == 1:
        w_tile = tl.join(codebook(low), codebook(high)) * scale[:, None]
    elif mode == 2:
        w_tile = tl.join(
            codebook(low) * scale[:, None], codebook(high) * scale[:, None]
        )
    else:
        w_tile = tl.join(codebook(low), codebook(high)) * scale[:, None, None]

    w_tile = tl.reshape(w_tile, (16, 32))
    tl.store(decoded_ptr + offs_n[:, None] * 32 + tl.arange(0, 32)[None, :], w_tile)

    x_tile = tl.full((16, 32), 1.0, dtype=tl.float32)
    acc = tl.zeros((16, 16), dtype=tl.float32)
    acc = tl.dot(x_tile, tl.trans(w_tile.to(x_tile.dtype)), acc=acc)
    tl.store(result_ptr + offs_m[:, None] * 16 + offs_n[None, :], acc)


def main():
    print(torch.cuda.get_device_properties(0), flush=True)
    for mode in range(4):
        decoded = torch.full((16, 32), 99.0, device="cuda", dtype=torch.float32)
        result = torch.full((16, 16), 99.0, device="cuda", dtype=torch.float32)
        weights = torch.full((16, 17), 0x11, device="cuda", dtype=torch.uint8)
        weights[:, 16] = 0x40
        probe[(1,)](decoded, result, weights, mode=mode, num_warps=1)
        torch.cuda.synchronize()
        print(
            "mode",
            mode,
            "decoded",
            decoded[0, :4].cpu().tolist(),
            "decoded_range",
            (decoded.min().item(), decoded.max().item()),
            "result",
            result[0, :4].cpu().tolist(),
            "result_range",
            (result.min().item(), result.max().item()),
            flush=True,
        )


if __name__ == "__main__":
    main()
