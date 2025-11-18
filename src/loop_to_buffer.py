from loop_ir import Function
from buffer_ir import Buffer, MemoryType


def lower_loop_to_buffer(func: Function) -> Buffer:
    if not func.outputs:
        raise ValueError("Function must have at least one output tensor.")

    output_tensor = func.outputs[0]

    return Buffer(
        name=output_tensor.name,
        tensor=output_tensor,
        memory=MemoryType.GLOBAL,
        body=func.body,
    )
