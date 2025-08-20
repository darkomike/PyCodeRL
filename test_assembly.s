.text
.globl main
main:
    nop   # Unsupported node: FunctionDef
    ret   # Return
    nop   # Unsupported node: Name
    call test_function  # Call test_function
    nop   # Unsupported node: Name
    add %rax, %rbx  # Addition
    nop   # Unsupported node: Name
    nop   # Unsupported node: Name
    nop   # Unsupported node: Constant
    nop   # Unsupported node: Constant
    nop   # Unsupported node: Name
    nop   # Unsupported node: Name
    movl $0, %eax  # Return 0
    ret
