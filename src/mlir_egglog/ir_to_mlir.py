from mlir_egglog.term_ir import Term as IRTerm
from mlir_egglog.mlir_gen import MLIRGen


def convert_term_to_mlir(tree: IRTerm, argspec: str) -> str:
    """
    Convert a term to MLIR.
    """

    argnames = map(lambda x: x.strip(), argspec.split(","))
    argmap = {k: f"%arg_{k}" for k in argnames}
    source = MLIRGen(tree, argmap).generate()
    return source
