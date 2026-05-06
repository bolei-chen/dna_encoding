from __future__ import annotations

from typing import Tuple

from reedsolo import RSCodec, ReedSolomonError


def pad_bits(dec: int) -> int:
    """
    Return the number of pad bits needed to reach a byte boundary.
    """
    if dec < 0:
        raise ValueError("decimal_value must be >= 0")
    bit_length = dec.bit_length()
    return (-bit_length) % 8


def rs_encode(bits: str, nsym: int) -> Tuple[str, int]:
    """
    RS-encode a bitstring. Returns (encoded_bits, pad_bits).
    We can evetually correct up to nsym // 2 bytes of errors.
    """
    if any(ch not in "01" for ch in bits):
        raise ValueError("bits must contain only '0' and '1'")
    if nsym < 0:
        raise ValueError("nsym must be >= 0")
    pad_bits = (-len(bits)) % 8
    if pad_bits:
        bits = f"{bits}{'0' * pad_bits}"

    data = int(bits, 2).to_bytes(len(bits) // 8, "big") if bits else b""
    if nsym == 0:
        encoded = data
    else:
        encoded = bytes(RSCodec(nsym).encode(data))

    encoded_bits = "".join(f"{byte:08b}" for byte in encoded)
    return encoded_bits, pad_bits


def rs_decode(encoded_bits: str, nsym: int, *, pad_bits: int = 0) -> str:
    """
    RS-decode a bitstring produced by `rs_encode`.
    """
    if any(ch not in "01" for ch in encoded_bits):
        raise ValueError("encoded_bits must contain only '0' and '1'")
    if len(encoded_bits) % 8 != 0:
        raise ValueError("encoded_bits length must be a multiple of 8")
    if nsym < 0:
        raise ValueError("nsym must be >= 0")

    data = int(encoded_bits, 2).to_bytes(len(encoded_bits) // 8, "big") if encoded_bits else b""
    if nsym == 0:
        decoded = data
    else:
        try:
            decoded, _, _ = RSCodec(nsym).decode(data)
        except ReedSolomonError as exc:
            raise ValueError("RS decode failed") from exc

    out_bits = "".join(f"{byte:08b}" for byte in decoded)
    return out_bits[:-pad_bits] if pad_bits else out_bits
