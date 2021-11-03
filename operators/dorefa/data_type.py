#!/usr/bin/env python
# coding=utf-8

class DataType:
    def __init__(self, bits, details, name=None):
        self.bits = bits
        self.lower = details[0][0]
        self.upper = details[0][1]
        self.max_val = details[1] - 1
        self.is_signed = self.lower < 0
        self.name = name


dorefa_uint2 = DataType(2, details=([0, 1], 2 ** 2), name="dorefa_uint2")
dorefa_uint4 = DataType(4, details=([0, 1], 2 ** 4), name="dorefa_uint4")
dorefa_uint8 = DataType(8, details=([0, 1], 2 ** 8), name="dorefa_uint8")
dorefa_uint16 = DataType(16, details=([0, 1], 2 ** 16), name="dorefa_uint16")
dorefa_int16 = DataType(16, details=([-0.5, 0.5], 2 ** 16), name="dorefa_int16")


DATA_TYPE_DICT = {"uint2": dorefa_uint2, "uint4": dorefa_uint4, "uint8": dorefa_uint8, "uint16": dorefa_uint16, "int16": dorefa_int16}
