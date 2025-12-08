# Directives

## Notation and Conventions

* Format is DIRECTIVE_NAME (opcode).
* Arguments can either be "hardcoded", meaning they are present in the sequence binary after the opcode, or "stack", meaning they are popped off the stack at runtime.
* Directives can have a "stack result type", which is the type that they push to the stack after execution.
* All multi-byte values are stored in **big-endian** byte order.
* `StackSizeType` is an alias for `U32` (4 bytes, unsigned).
* `len(stack)` refers to the current number of bytes on the stack.
* `max_stack_size` is the maximum allowed stack size (implementation-defined, e.g., 4096 bytes).
* Array slice notation `stack[a .. b)` means bytes from index `a` (inclusive) to `b` (exclusive).

## Stack Frame Model

The sequencer maintains two pointers:
- `next_dir_idx`: Index of the next directive to execute.
- `stack_frame_start`: Byte offset from the start of the stack where the current function's local variables begin.

**Stack Layout During Function Call:**
```
Low addresses                                                        High addresses
[global vars][...caller locals...][args][ret_addr][saved_fp][callee locals][top]
                                                            ^
                                             stack_frame_start
```

- **Global variables** occupy a fixed region at the bottom of the stack (starting at offset 0).
- **Function arguments** are pushed by the caller before CALL and accessed by the callee using negative offsets from `stack_frame_start`.
- **Stack frame header** consists of 8 bytes: `return_addr` (U32) followed by `saved_frame_ptr` (U32). This is `STACK_FRAME_HEADER_SIZE = 8`.
- **Local variables** are accessed using non-negative offsets from `stack_frame_start`.

## Error Codes

The following error codes may be returned by directives:
- `STACK_ACCESS_OUT_OF_BOUNDS`: Attempted to read/write outside valid stack region.
- `STACK_OVERFLOW`: Operation would exceed `max_stack_size`.
- `STMT_OUT_OF_BOUNDS`: Jump target is outside the directive array.
- `DOMAIN_ERROR`: Mathematical operation on invalid input (e.g., division by zero).

---

## WAIT_REL (1)
Sleeps for a relative duration from the current time.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| useconds  | U32      | stack  | Wait time in microseconds (must be less than a second) |
| seconds  | U32      | stack  | Wait time in seconds |

## WAIT_ABS (2)
Sleeps until an absolute time.
| Arg Name      | Arg Type | Source | Description |
|---------------|----------|--------|-------------|
| useconds     | U32      | stack  | Microseconds |
| seconds      | U32      | stack  | Seconds |
| time_context | FwTimeContextStoreType       | stack  | Time context (user defined value, unused by Fpy) |
| time_base    | U16      | stack  | Time base |

## GOTO (3)
Sets the index of the next directive to execute.
| Arg Name | Arg Type | Source     | Description |
|----------|----------|------------|-------------|
| dir_idx  | U32      | hardcoded | The statement index to execute next |

## IF (4)
Pops a byte off the stack. If the byte is not 0, proceed to the next directive, otherwise goto a hardcoded directive index.
 
| Arg Name             | Arg Type | Source     | Description |
|---------------------|----------|------------|-------------|
| false_goto_dir_index| U32      | hardcoded | Directive index to jump to if false |
| condition          | bool     | stack     | Condition to evaluate |

## NO_OP (5)
Does nothing.

## PUSH_TLM_VAL (6)
Pushes a telemetry value buffer to the stack.
| Arg Name     | Arg Type | Source     | Description |
|--------------|----------|------------|-------------|
| chan_id      | U32      | hardcoded | the tlm channel id to get the time of |

| Stack Result Type | Description |
| ------------------|-------------|
| bytes | The raw bytes of the telemetry value buffer |
## PUSH_PRM (7)
Pushes a parameter buffer to the stack.
| Arg Name     | Arg Type | Source     | Description |
|--------------|----------|------------|-------------|
| prm_id       | U32      | hardcoded | the param id to get the value of |

| Stack Result Type | Description |
| ------------------|-------------|
| bytes | The raw bytes of the parameter buffer |

## CONST_CMD (8)
Runs a command with a constant opcode and a constant byte array of arguments.
| Arg Name   | Arg Type | Source     | Description |
|------------|----------|------------|-------------|
| cmd_opcode | U32      | hardcoded | Command opcode |
| args       | bytes    | hardcoded | Command arguments |

| Stack Result Type | Description |
| ------------------|-------------|
| Fw.CmdResponse | The CmdResponse that the command returned |


## OR (9)
Performs an `or` between two booleans, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | bool     | stack  | Right operand |
| lhs      | bool     | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| bool | The result |

## AND (10)
Performs an `and` between two booleans, pushes result to stack.

| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | bool     | stack  | Right operand |
| lhs      | bool     | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| bool | The result |
## IEQ (11)
Compares two integers for equality, pushes result to stack. Doesn't differentiate between signed and unsigned.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | I64      | stack  | Right operand |
| lhs      | I64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| bool | The result |
## INE (12)
Compares two integers for inequality, pushes result to stack. Doesn't differentiate between signed and unsigned.

| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | I64      | stack  | Right operand |
| lhs      | I64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| bool | The result |
## ULT (13)
Performs an unsigned less than comparison on two unsigned integers, pushes result to stack
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | U64      | stack  | Right operand |
| lhs      | U64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| bool | The result |
## ULE (14)
Performs an unsigned less than or equal to comparison on two unsigned integers, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | U64      | stack  | Right operand |
| lhs      | U64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| bool | The result |
## UGT (15)
Performs an unsigned greater than comparison on two unsigned integers, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | U64      | stack  | Right operand |
| lhs      | U64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| bool | The result |
## UGE (16)
Performs an unsigned greater than or equal to comparison on two unsigned integers, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | U64      | stack  | Right operand |
| lhs      | U64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| bool | The result |
## SLT (17)
Performs a signed less than comparison on two signed integers, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | I64      | stack  | Right operand |
| lhs      | I64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| bool | The result |
## SLE (18)
Performs a signed less than or equal to comparison on two signed integers, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | I64      | stack  | Right operand |
| lhs      | I64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| bool | The result |
## SGT (19)
Performs a signed greater than comparison on two signed integers, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | I64      | stack  | Right operand |
| lhs      | I64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| bool | The result |
## SGE (20)
Performs a signed greater than or equal to comparison on two signed integers, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | I64      | stack  | Right operand |
| lhs      | I64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| bool | The result |
## FEQ (21)
Compares two floats for equality, pushes result to stack. If neither is NaN and they are otherwise equal, pushes 1 to stack, otherwise 0. Infinity is handled consistent with C++.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | F64      | stack  | Right operand |
| lhs      | F64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| bool | The result |
## FNE (22)
Compares two floats for inequality, pushes result to stack. If either is NaN or they are not equal, pushes 1 to stack, otherwise 0. Infinity is handled consistent with C++.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | F64      | stack  | Right operand |
| lhs      | F64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| bool | The result |
## FLT (23)
Performs a less than comparison on two floats, pushes result to stack. If neither is NaN and the second < first, pushes 1 to stack, otherwise 0. Infinity is handled consistent with C++.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | F64      | stack  | Right operand |
| lhs      | F64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| bool | The result |
## FLE (24)
Performs a less than or equal to comparison on two floats, pushes result to stack. If neither is NaN and the second <= first, pushes 1 to stack, otherwise 0. Infinity is handled consistent with C++.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | F64      | stack  | Right operand |
| lhs      | F64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| bool | The result |
## FGT (25)
Performs a greater than comparison on two floats, pushes result to stack. If neither is NaN and the second > first, pushes 1 to stack, otherwise 0. Infinity is handled consistent with C++.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | F64      | stack  | Right operand |
| lhs      | F64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| bool | The result |
## FGE (26)
Performs a greater than or equal to comparison on two floats, pushes result to stack. If neither is NaN and the second >= first, pushes 1 to stack, otherwise 0. Infinity is handled consistent with C++.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | F64      | stack  | Right operand |
| lhs      | F64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| bool | The result |
## NOT (27)
Performs a boolean not operation on a boolean, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| value    | bool     | stack  | Value to negate |

| Stack Result Type | Description |
| ------------------|-------------|
| bool | The result |
## FPTOSI (28)
Converts a float to a signed integer, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| value    | F64      | stack  | Float to convert |

| Stack Result Type | Description |
| ------------------|-------------|
| I64 | The result |
## FPTOUI (29)
Converts a float to an unsigned integer, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| value    | F64      | stack  | Float to convert |

| Stack Result Type | Description |
| ------------------|-------------|
| U64 | The result |
## SITOFP (30)
Converts a signed integer to a float, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| value    | I64      | stack  | Integer to convert |
| Stack Result Type | Description |
| ------------------|-------------|
| F64 | The result |
## UITOFP (31)
Converts an unsigned integer to a float, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| value    | U64      | stack  | Integer to convert |

| Stack Result Type | Description |
| ------------------|-------------|
| F64 | The result |
## ADD (32)
Performs integer addition, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | I64      | stack  | Right operand |
| lhs      | I64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| I64 | The result |
## SUB (33)
Performs integer subtraction, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | I64      | stack  | Right operand |
| lhs      | I64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| I64 | The result |
## MUL (34)
Performs integer multiplication, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | I64      | stack  | Right operand |
| lhs      | I64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| I64 | The result |
## UDIV (35)
Performs unsigned integer division, pushes result to stack. A divisor of 0 will result in DOMAIN_ERROR.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | U64      | stack  | Right operand |
| lhs      | U64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| U64 | The result |
## SDIV (36)
Performs signed integer division, pushes result to stack. A divisor of 0 will result in DOMAIN_ERROR.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | I64      | stack  | Right operand |
| lhs      | I64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| I64 | The result |
## UMOD (37)
Performs unsigned integer modulo, pushes result to stack. A 0 divisor (rhs) will result in DOMAIN_ERROR.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | U64      | stack  | Right operand |
| lhs      | U64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| U64 | The result |
## SMOD (38)
Performs signed integer modulo, pushes result to stack. A 0 divisor (rhs) will result in DOMAIN_ERROR.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | I64      | stack  | Right operand |
| lhs      | I64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| I64 | The result |
## FADD (39)
Performs float addition, pushes result to stack. NaN, and infinity are handled consistently with C++ addition.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | F64      | stack  | Right operand |
| lhs      | F64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| F64 | The result |
## FSUB (40)
Performs float subtraction, pushes result to stack. NaN, and infinity are handled consistently with C++ subtraction.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | F64      | stack  | Right operand |
| lhs      | F64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| F64 | The result |
## FMUL (41)
Performs float multiplication, pushes result to stack. NaN, and infinity are handled consistently with C++ multiplication.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | F64      | stack  | Right operand |
| lhs      | F64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| F64 | The result |
## FDIV (42)
Performs float division, pushes result to stack. Zero divisors, NaN, and infinity are handled consistently with C++ division.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | F64      | stack  | Right operand |
| lhs      | F64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| F64 | The result |

## FPOW (43)
Performs float exponentiation, pushes result to stack. NaN and infinity values are handled consistently with C++ `std::pow`.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| exp      | F64      | stack  | Exponent value |
| base     | F64      | stack  | Base value |

| Stack Result Type | Description |
| ------------------|-------------|
| F64 | The result |
## FLOG (44)
Performs float logarithm, pushes result to stack. Negatives yield a DOMAIN_ERROR, NaN and infinity values are handled consistently with C++ `std::log`.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| value    | F64      | stack  | Value for logarithm |

| Stack Result Type | Description |
| ------------------|-------------|
| F64 | The result |
## FMOD (45)
Performs float modulo, pushes result to stack. A 0 divisor (rhs) will result in a DOMAIN_ERROR. A NaN will produce a NaN result or infinity as either argument yields NaN.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| rhs      | F64      | stack  | Right operand |
| lhs      | F64      | stack  | Left operand |

| Stack Result Type | Description |
| ------------------|-------------|
| F64 | The result |
## FPEXT (46)
Extends a 32-bit float to a 64-bit float, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| value    | F32      | stack  | Float to extend |

| Stack Result Type | Description |
| ------------------|-------------|
| F64 | The result |

## FPTRUNC (47)
Truncates a 64-bit float to a 32-bit float, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| value    | F64      | stack  | Value to truncate |

| Stack Result Type | Description |
| ------------------|-------------|
| F32 | The result |

## SIEXT_8_64 (48)
Sign-extends an 8-bit integer to a 64-bit integer, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| value    | I8       | stack  | Value to extend |

| Stack Result Type | Description |
| ------------------|-------------|
| I64 | The result |

## SIEXT_16_64 (49)
Sign-extends a 16-bit integer to a 64-bit integer, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| value    | I16      | stack  | Value to extend |

| Stack Result Type | Description |
| ------------------|-------------|
| I64 | The result |
## SIEXT_32_64 (50)
Sign-extends a 32-bit integer to a 64-bit integer, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| value    | I32      | stack  | Value to extend |

| Stack Result Type | Description |
| ------------------|-------------|
| I64 | The result |
## ZIEXT_8_64 (51)
Zero-extends an 8-bit integer to a 64-bit integer, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| value    | U8       | stack  | Value to extend |

| Stack Result Type | Description |
| ------------------|-------------|
| U64 | The result |
## ZIEXT_16_64 (52)
Zero-extends a 16-bit integer to a 64-bit integer, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| value    | U16      | stack  | Value to extend |

| Stack Result Type | Description |
| ------------------|-------------|
| U64 | The result |
## ZIEXT_32_64 (53)
Zero-extends a 32-bit integer to a 64-bit integer, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| value    | U32      | stack  | Value to extend |

| Stack Result Type | Description |
| ------------------|-------------|
| U64 | The result |
## ITRUNC_64_8 (54)
Truncates a 64-bit integer to an 8-bit integer, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| value    | I64      | stack  | Value to truncate |

| Stack Result Type | Description |
| ------------------|-------------|
| U8 | The result |
## ITRUNC_64_16 (55)
Truncates a 64-bit integer to a 16-bit integer, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| value    | I64      | stack  | Value to truncate |

| Stack Result Type | Description |
| ------------------|-------------|
| I16 | The result |
## ITRUNC_64_32 (56)
Truncates a 64-bit integer to a 32-bit integer, pushes result to stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| value    | I64      | stack  | Value to truncate |

| Stack Result Type | Description |
| ------------------|-------------|
| I32 | The result |

## EXIT (57)
Pops a byte off the stack. If the byte == 0, end sequence as if it had finished nominally, otherwise exit the sequence and raise an event with an error code.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| success    | U8      | stack  | 0 if should exit without error |


## ALLOCATE (58)
Pushes a hard-coded count of 0x00-bytes to the stack.
| Arg Name | Arg Type | Source     | Description |
|----------|----------|------------|-------------|
| size     | U32      | hardcoded  | Bytes to allocate |

| Stack Result Type | Description |
| ------------------|-------------|
| bytes | A series of 0 bytes of length `size` |

## STORE_LOCAL_CONST_OFFSET (59)
Stores a value to a local variable at a compile-time-known offset relative to the current stack frame.

**Preconditions:**
- `len(stack) >= size`
- `stack_frame_start + lvar_offset >= 0`
- `stack_frame_start + lvar_offset + size <= len(stack)`

**Semantics:**
1. Let `value` be the top `size` bytes of the stack (big-endian, with the first byte at `stack[len(stack) - size]`).
2. Remove these `size` bytes from the stack.
3. Write `value` to `stack[stack_frame_start + lvar_offset .. stack_frame_start + lvar_offset + size)`.

**Error Conditions:**
- If `len(stack) < size`: `STACK_ACCESS_OUT_OF_BOUNDS`
- If `stack_frame_start + lvar_offset < 0`: `STACK_ACCESS_OUT_OF_BOUNDS`
- If `stack_frame_start + lvar_offset + size > len(stack)` (after pop): `STACK_ACCESS_OUT_OF_BOUNDS`

| Arg Name    | Arg Type | Source     | Description |
|-------------|----------|------------|-------------|
| lvar_offset | I32      | hardcoded  | Signed byte offset relative to `stack_frame_start`. Negative values access memory below the frame (e.g., function arguments). |
| size        | U32      | hardcoded  | Number of bytes to store. |
| value       | bytes    | stack      | The value to store (popped from stack top). |

## LOAD_LOCAL (60)
Loads a value from a local variable at a compile-time-known offset relative to the current stack frame, and pushes it to the stack.

**Preconditions:**
- `stack_frame_start + lvar_offset >= 0`
- `stack_frame_start + lvar_offset + size <= len(stack)`
- `len(stack) + size <= max_stack_size`

**Semantics:**
1. Let `addr = stack_frame_start + lvar_offset`.
2. Read `size` bytes from `stack[addr .. addr + size)`.
3. Push these bytes to the top of the stack (preserving byte order).

**Error Conditions:**
- If `stack_frame_start + lvar_offset < 0`: `STACK_ACCESS_OUT_OF_BOUNDS`
- If `stack_frame_start + lvar_offset + size > len(stack)`: `STACK_ACCESS_OUT_OF_BOUNDS`
- If `len(stack) + size > max_stack_size`: `STACK_OVERFLOW`

| Arg Name    | Arg Type | Source     | Description |
|-------------|----------|------------|-------------|
| lvar_offset | I32      | hardcoded  | Signed byte offset relative to `stack_frame_start`. Negative values access memory below the frame (e.g., function arguments pushed before CALL). |
| size        | U32      | hardcoded  | Number of bytes to load. |

| Stack Result Type | Description |
| ------------------|-------------|
| bytes | The `size` bytes read from the local variable location. |

## PUSH_VAL (61)
Pushes a constant array of bytes to the stack.
| Arg Name | Arg Type | Source     | Description |
|----------|----------|------------|-------------|
| val      | bytes    | hardcoded  | the byte array to push |

| Stack Result Type | Description |
| ------------------|-------------|
| bytes | The byte array from the arg |

## DISCARD (62)
Discards bytes from the top of the stack.
| Arg Name | Arg Type | Source     | Description |
|----------|----------|------------|-------------|
| size     | U32      | hardcoded  | Bytes to discard |


## MEMCMP (63)
Compares two memory regions on the stack.
| Arg Name | Arg Type | Source     | Description |
|----------|----------|------------|-------------|
| size     | U32      | hardcoded  | Bytes to compare |

| Stack Result Type | Description |
| ------------------|-------------|
| bool | The result |

## STACK_CMD (64)
Dispatches a command with arguments from the stack.
| Arg Name  | Arg Type | Source     | Description |
|-----------|----------|------------|-------------|
| args_size | U32      | hardcoded  | Size of command arguments |

| Stack Result Type | Description |
| ------------------|-------------|
| Fw.CmdResponse | The CmdResponse that the command returned |

## PUSH_TLM_VAL_AND_TIME (65)
Gets a telemetry channel and pushes its value, and then its time, onto the stack.
| Arg Name     | Arg Type | Source     | Description |
|--------------|----------|------------|-------------|
| chan_id      | U32      | hardcoded | the tlm channel id to get |

| Stack Result Type | Description |
| ------------------|-------------|
| bytes | The raw bytes of the telemetry value buffer |
| Fw.Time | The time tag of the telemetry value |

## PUSH_TIME (66)
Pushes the current time, from the `timeCaller` port, to the stack.
| Stack Result Type | Description |
| ------------------|-------------|
| Fw.Time | The current time |

## SET_FLAG (67)
Pops a bool off the stack, sets a flag with a specific index to that bool.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| flag_idx | U8 | hardcoded | Index of the flag to set |
| value | bool | stack | Value to set the flag to |

## GET_FLAG (68)
Gets a flag and pushes its value as a U8 to the stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| flag_idx | U8 | hardcoded | Index of the flag to get |

| Stack Result Type | Description |
| ------------------|-------------|
| bool | The value of the flag |

## GET_FIELD (69)
Pops an offset (StackSizeType) off the stack. Takes a hard-coded number of bytes from top of stack, and then inside of that a second array of hard-coded number of bytes. The second array is offset by the value previously popped off the stack, with offset 0 meaning the second array starts furthest down the stack. Leaves only the second array of bytes, deleting the surrounding bytes.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| parent_size | U32 | hardcoded | Size of the struct |
| member_size | U32 | hardcoded | Size of the field |
| offset | U32 | stack | Offset of the field in the struct |

| Stack Result Type | Description |
| ------------------|-------------|
| bytes | The raw bytes of the field |

## PEEK (70)
Pops a StackSizeType `offset` off the stack, then a StackSizeType `byteCount`. Let `top` be the top of the stack. Takes the region starting at `top - offset - byteCount` and going to `top - offset`, and pushes this region to the top of the stack.
| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| offset | StackSizeType | stack | Offset from top at which to peek |
| byteCount | StackSizeType | stack | Number of bytes to peek at, starting at offset, and going downwards in the stack |

| Stack Result Type | Description |
| ------------------|-------------|
| bytes | The peeked bytes |

## STORE_LOCAL (71)
Stores a value to a local variable at a runtime-determined offset relative to the current stack frame.

**Preconditions:**
- `len(stack) >= size + 4` (value bytes + offset)
- After popping offset: `stack_frame_start + lvar_offset >= 0`
- After popping offset: `stack_frame_start + lvar_offset + size <= len(stack)`

**Semantics:**
1. Pop a 4-byte signed integer `lvar_offset` from the stack (big-endian, I32).
2. Let `value` be the top `size` bytes of the remaining stack.
3. Remove these `size` bytes from the stack.
4. Let `addr = stack_frame_start + lvar_offset`.
5. Write `value` to `stack[addr .. addr + size)`.

**Error Conditions:**
- If `len(stack) < size + 4`: `STACK_ACCESS_OUT_OF_BOUNDS`
- If `stack_frame_start + lvar_offset < 0`: `STACK_ACCESS_OUT_OF_BOUNDS`
- If `stack_frame_start + lvar_offset + size > len(stack)` (after popping offset, before popping value): `STACK_ACCESS_OUT_OF_BOUNDS`

| Arg Name    | Arg Type | Source     | Description |
|-------------|----------|------------|-------------|
| size        | U32      | hardcoded  | Number of bytes to store. |
| lvar_offset | I32      | stack      | Signed byte offset relative to `stack_frame_start`. |
| value       | bytes    | stack      | The value to store (below the offset on stack). |

## CALL (72)
Performs a function call. Pops the target directive index from the stack, saves the return address and current frame pointer to the stack, then transfers control to the target.

**Preconditions:**
- `len(stack) >= 4` (for target address)
- `len(stack) + 8 <= max_stack_size` (space for return address and frame pointer)
- `0 <= target < len(directives)` (checked at jump time by main loop)

**Semantics (in order):**
1. Pop a 4-byte unsigned integer `target` from the stack (StackSizeType = U32, big-endian).
2. Let `return_addr = next_dir_idx` (the index of the instruction that would execute after this CALL).
3. Set `next_dir_idx = target`.
4. Push `return_addr` as a 4-byte unsigned integer (U32, big-endian) to the stack.
5. Push `stack_frame_start` as a 4-byte unsigned integer (U32, big-endian) to the stack.
6. Set `stack_frame_start = len(stack)` (the new frame begins immediately after the saved frame pointer).

**Stack Layout After CALL:**
```
[... function arguments ...][return_addr (4 bytes)][saved_frame_ptr (4 bytes)]
                                                                              ^
                                                        stack_frame_start ────┘
```

**Error Conditions:**
- If `len(stack) < 4`: `STACK_ACCESS_OUT_OF_BOUNDS`
- If `len(stack) + 8 > max_stack_size`: `STACK_OVERFLOW`
- If `target >= len(directives)`: `STMT_OUT_OF_BOUNDS` (checked when directive executes)

**Note:** Function arguments must be pushed to the stack before the target address. The callee accesses arguments using negative `lvar_offset` values in LOAD_LOCAL (e.g., `lvar_offset = -(STACK_FRAME_HEADER_SIZE + arg_size)` where `STACK_FRAME_HEADER_SIZE = 8`).

| Arg Name | Arg Type | Source | Description |
|----------|----------|--------|-------------|
| target   | U32 (StackSizeType) | stack | Directive index to jump to. |

## RETURN (73)
Returns from a function call. Restores the caller's execution context and optionally returns a value.

**Preconditions:**
- `len(stack) >= return_val_size` (for return value, if any)
- `len(stack) >= stack_frame_start` (sanity check)
- After truncating to frame: `len(stack) >= 8` (saved frame pointer + return address)
- After restoring frame: `len(stack) >= call_args_size` (to discard arguments)

**Semantics (in order):**
1. If `return_val_size > 0`: Pop `return_val_size` bytes from the stack as `return_value`.
2. Truncate the stack to `stack_frame_start` (discard all local variables allocated in this frame).
3. Pop a 4-byte unsigned integer as `saved_frame_ptr` (U32, big-endian).
4. Pop a 4-byte unsigned integer as `return_addr` (U32, big-endian).
5. Set `stack_frame_start = saved_frame_ptr`.
6. Set `next_dir_idx = return_addr`.
7. Pop and discard `call_args_size` bytes (the function arguments pushed by the caller).
8. If `return_val_size > 0`: Push `return_value` to the stack.

**Stack Transformation:**
```
Before RETURN:
[... caller locals ...][args (call_args_size)][ret_addr][saved_fp][... callee locals ...][return_value]
                                                                  ^                       ^
                                                   stack_frame_start                      stack top

After RETURN:
[... caller locals ...][return_value]
^                                    ^
stack_frame_start (restored)         stack top
```

**Error Conditions:**
- If `len(stack) < return_val_size`: `STACK_ACCESS_OUT_OF_BOUNDS`
- If `stack_frame_start > len(stack)`: `STACK_ACCESS_OUT_OF_BOUNDS` (corrupt frame)
- If remaining stack after truncation `< 8`: `STACK_ACCESS_OUT_OF_BOUNDS`
- If remaining stack after header pop `< call_args_size`: `STACK_ACCESS_OUT_OF_BOUNDS`

| Arg Name        | Arg Type | Source     | Description |
|-----------------|----------|------------|-------------|
| return_val_size | U32      | hardcoded  | Size of return value in bytes. Use 0 for void functions. |
| call_args_size  | U32      | hardcoded  | Total size of function arguments in bytes. This must match the bytes pushed by the caller before CALL. |

| Stack Result Type | Description |
| ------------------|-------------|
| bytes | The return value (only if `return_val_size > 0`). |

## LOAD_GLOBAL (74)
Loads a value from an absolute address in the stack (used for global variables), and pushes it to the stack.

**Preconditions:**
- `global_offset >= 0`
- `global_offset + size <= len(stack)`
- `len(stack) + size <= max_stack_size`

**Semantics:**
1. Read `size` bytes from `stack[global_offset .. global_offset + size)`.
2. Push these bytes to the top of the stack (preserving byte order).

**Error Conditions:**
- If `global_offset < 0`: `STACK_ACCESS_OUT_OF_BOUNDS`
- If `global_offset + size > len(stack)`: `STACK_ACCESS_OUT_OF_BOUNDS`
- If `len(stack) + size > max_stack_size`: `STACK_OVERFLOW`

| Arg Name      | Arg Type | Source     | Description |
|---------------|----------|------------|-------------|
| global_offset | U32      | hardcoded  | Absolute byte offset from the start of the stack (index 0). |
| size          | U32      | hardcoded  | Number of bytes to load. |

| Stack Result Type | Description |
| ------------------|-------------|
| bytes | The `size` bytes read from the global variable location. |

## STORE_GLOBAL (75)
Stores a value to an absolute address in the stack (used for global variables), with the offset determined at runtime.

**Preconditions:**
- `len(stack) >= size + 4` (value bytes + offset)
- After popping offset: `global_offset >= 0`
- After popping offset: `global_offset + size <= len(stack)`

**Semantics:**
1. Pop a 4-byte signed integer `global_offset` from the stack (big-endian, I32).
2. Let `value` be the top `size` bytes of the remaining stack.
3. Remove these `size` bytes from the stack.
4. Write `value` to `stack[global_offset .. global_offset + size)`.

**Error Conditions:**
- If `len(stack) < size + 4`: `STACK_ACCESS_OUT_OF_BOUNDS`
- If `global_offset < 0`: `STACK_ACCESS_OUT_OF_BOUNDS`
- If `global_offset + size > len(stack)` (after popping offset, before popping value): `STACK_ACCESS_OUT_OF_BOUNDS`

| Arg Name      | Arg Type | Source     | Description |
|---------------|----------|------------|-------------|
| size          | U32      | hardcoded  | Number of bytes to store. |
| global_offset | I32      | stack      | Absolute byte offset from the start of the stack. |
| value         | bytes    | stack      | The value to store (below the offset on stack). |

## STORE_GLOBAL_CONST_OFFSET (76)
Stores a value to an absolute address in the stack (used for global variables), with a compile-time-known offset.

**Preconditions:**
- `len(stack) >= size`
- `global_offset >= 0`
- `global_offset + size <= len(stack)`

**Semantics:**
1. Let `value` be the top `size` bytes of the stack.
2. Remove these `size` bytes from the stack.
3. Write `value` to `stack[global_offset .. global_offset + size)`.

**Error Conditions:**
- If `len(stack) < size`: `STACK_ACCESS_OUT_OF_BOUNDS`
- If `global_offset < 0`: `STACK_ACCESS_OUT_OF_BOUNDS`
- If `global_offset + size > len(stack)` (after pop): `STACK_ACCESS_OUT_OF_BOUNDS`

| Arg Name      | Arg Type | Source     | Description |
|---------------|----------|------------|-------------|
| global_offset | U32      | hardcoded  | Absolute byte offset from the start of the stack. |
| size          | U32      | hardcoded  | Number of bytes to store. |
| value         | bytes    | stack      | The value to store (popped from stack top). |
